# RTMP 推流模組

import cv2
import sys
from config import ip_addr, g_width

# 推流
def push_stream(push_queue, process, timer, thread_id, shared):
    
    counter = 0
    second = 0
    minute = 0
    try:
        while shared.running.value:
            if push_queue.empty(): continue
            # 顯示分數
            if timer.should_wait(thread_id): continue
            
            counter += 1
            if counter == 15:
                counter = 0
                second += 1
            if second == 60:
                second = 0
                minute += 1

            encoded_frame = push_queue.get()
            frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
            with shared.player1_bodyhit.get_lock():
                p1_body = shared.player1_bodyhit.value
            with shared.player2_bodyhit.get_lock():
                p2_body = shared.player2_bodyhit.value
            with shared.player1_headhit.get_lock():
                p1_head = shared.player1_headhit.value
            with shared.player2_headhit.get_lock():
                p2_head = shared.player2_headhit.value

            cv2.putText(frame, f"red_player_body:{p1_body}", (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"red_player_head:{p1_head}", (0,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"blue_player_body:{p2_body}", (0,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"blue_player_head:{p2_head}", (0,110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"{minute}:{second}", (int(g_width/2),20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            process.stdin.write(frame.tobytes())
            timer.increment(thread_id)
    except KeyboardInterrupt:
        sys.exit(0)
