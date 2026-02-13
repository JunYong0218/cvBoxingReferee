#主程式

import cv2
import multiprocessing
from multiprocessing import Process, Queue, Barrier
import threading
import subprocess
import signal
import sys

from config import SharedVars, ThreadTimer, ip_addr, g_width, g_height
from proces import get_frame, cam, mix_score
from streamer import push_stream

def main():
    def signal_handler(signum, frame):
        print("\n接收到中斷信號，正在清理...")
        shared.running.value = 0
        
        # 等待所有進程
        for p in processes:
            if p.is_alive():
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
        
        cv2.destroyAllWindows()
        sys.exit(0)

    # 註冊信號
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    shared = SharedVars()
    barrier = Barrier(4)

    try:
        game_id = int(input("請輸入比賽 ID: "))
        print(f"比賽 ID: {game_id}")

        ffmpeg_cmd1 = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{g_width}x{g_height}',
        '-pix_fmt', 'bgr24',
        '-r', '15',
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-pix_fmt', 'yuv420p',
        '-f', 'flv',
        '-flvflags', 'no_duration_filesize',
        f'rtmp://{ip_addr}/hls/{game_id}-1']
    
        ffmpeg_process1 = subprocess.Popen(ffmpeg_cmd1, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        ffmpeg_cmd2 = [
                'ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{g_width}x{g_height}',
                '-pix_fmt', 'bgr24',
                '-r', '15',
                '-i', '-',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-pix_fmt', 'yuv420p',
                '-f', 'flv',
                '-flvflags', 'no_duration_filesize',
                f'rtmp://{ip_addr}/hls/{game_id}-2']
            
        ffmpeg_process2 = subprocess.Popen(ffmpeg_cmd2, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        ffmpeg_cmd3 = [
                'ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{g_width}x{g_height}',
                '-pix_fmt', 'bgr24',
                '-r', '15',
                '-i', '-',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-pix_fmt', 'yuv420p',
                '-f', 'flv',
                '-flvflags', 'no_duration_filesize',
                f'rtmp://{ip_addr}/hls/{game_id}-3']
            
        ffmpeg_process3 = subprocess.Popen(ffmpeg_cmd3, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        # 建立隊列
        cam1_queue = Queue()
        cam2_queue = Queue()    
        cam3_queue = Queue()    
        image1_queue = Queue()
        image2_queue = Queue()  
        image3_queue = Queue()  
        video_queue1 = Queue()
        video_queue2 = Queue()  
        video_queue3 = Queue()  
        push_queue1 = Queue()
        push_queue2 = Queue()   
        push_queue3 = Queue()   

        timer = ThreadTimer()
        # 建立處理程序
        processes = []
        p1 = Process(target=cam, args=(video_queue1, cam1_queue, image1_queue, 0, shared, push_queue1))
        p2 = Process(target=cam, args=(video_queue2, cam2_queue, image2_queue, 1, shared, push_queue2))
        p3 = Process(target=cam, args=(video_queue3, cam3_queue, image3_queue, 2, shared, push_queue3))

        p5 = Process(target=mix_score, args=(cam1_queue, cam2_queue, cam3_queue, 
                                           image1_queue, image2_queue, image3_queue,
                                           barrier, shared, game_id))
        p6 = Process(target=get_frame,args=(0, video_queue1, barrier, 3, shared))
        p7 = Process(target=get_frame,args=(1, video_queue2, barrier, 4, shared))
        p8 = Process(target=get_frame,args=(2, video_queue3, barrier, 5, shared))

        threads = []
        t1 = threading.Thread(target=push_stream, args=(push_queue1, ffmpeg_process1, timer, 0, shared))
        t2 = threading.Thread(target=push_stream, args=(push_queue2, ffmpeg_process2, timer, 1, shared))
        t3 = threading.Thread(target=push_stream, args=(push_queue3, ffmpeg_process3, timer, 2, shared))
        
        processes.extend([p1, p2, p3, p5, p6, p7, p8])
        threads.extend([t1,t2,t3])
        # 啟動所有處理程序
        for p in processes:
            p.start()
        # 啟動所有線程
        for t in threads:
            t.start()

        # 等待處理程序結束
        for p in processes:
            p.join()
        # 等待線程結束
        for t in threads:
            t.join()
            
    except Exception as e:
        print(f"Error in main: {e}")
        shared.running.value = 0
        for p in processes:
            if p.is_alive():
                p.terminate()
        return 1
    finally:
        if ffmpeg_process1 and ffmpeg_process2 and ffmpeg_process3:
            ffmpeg_process = []
            ffmpeg_process.extend([ffmpeg_process1,ffmpeg_process2,ffmpeg_process3])
            for f in ffmpeg_process:
                f.terminate()
                f.wait()
    return 0

if __name__ == "__main__":
    multiprocessing.freeze_support()
    exit(main())
