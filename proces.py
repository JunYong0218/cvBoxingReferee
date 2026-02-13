"""
畫面處理模組 - 從原始 identification.py 直接複製所有處理函數
完全保持原始邏輯、變數命名、數值不變
"""

import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon
from shapely.geometry.collection import GeometryCollection
import asyncio
import websockets
import base64
import json
import pymysql
from queue import Empty
import sys
import os

from config import ip_addr, g_width, g_height
from core_worker import init_model

# WebSocket 發送
async def send_image(payload):
    uri = f"ws://{ip_addr}:3005"
    async with websockets.connect(uri) as websocket:
        await websocket.send(payload)
        print("socket sended")

# 顏色分析函數
def get_dominant_hue(frame, polygon):
    """
    計算多邊形範圍內的主導色相 (Hue)。
    """
    """if polygon == None:
        return None"""
    
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    polygon_pixels = frame[mask == 255]
    
    if len(polygon_pixels) == 0:
        return None

    hsv_pixels = cv2.cvtColor(polygon_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    
    if len(hsv_pixels) == 0:
        return None

    h_values = hsv_pixels[:, 0]
    hist, bins = np.histogram(h_values, bins=180, range=[0, 180])
    dominant_hue = bins[np.argmax(hist)]

    return dominant_hue

# 確保多邊形有效
def is_valid_polygon(polygon_coords):
    if polygon_coords is None or len(polygon_coords) < 3:
        return False
    try:
        polygon = Polygon(polygon_coords)
        return polygon.is_valid
    except Exception as e:
        print(f"Invalid polygon: {e}")
        return False

# 攝影機讀取
def get_frame(video_path, video_queue, barrier, process_id, shared):
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, g_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, g_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*"MJPG"))
        count = 1
        print(f"Process {process_id} starting")
        while cap.isOpened() and shared.running.value:
            
            ret, frame = cap.read()
            if count==1:  barrier.wait()
            count+=1
            if not ret:
                break
            if (count % 2) == 0:
                continue
            encoded_image = cv2.imencode('.jpg', frame)[1]
            video_queue.put(encoded_image)
        cap.release()
    except KeyboardInterrupt:
        os._exit(0)

# 辨識主處理
def cam(video_queue, queue, image_queue, process_id, shared, push_queue):
    print(f"Process {process_id} waiting at barrier")
    
    try:
        # 在處理程序中初始化模型
        model = init_model()
        dominant_color = []
        while shared.running.value:
            if video_queue.empty():
                continue
            encoded_frame = video_queue.get()
            frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
            ###
            #real 要注意
            ###
            org_frame = frame
            
            width = g_width
            height = g_height
            
            #with shared.model_lock: #這是改過的模型鎖
            results = model.predict(source=frame, save=False, save_txt=False, verbose=False)
            
            result = results[0]
            current_masks = []
            gloves_color = []
            head_poly = [None] * 2
            boxer_poly = [None] * 2
            gloves_poly = [None] * 4
            sta_list = []
            segmentation_contours_idx = []
            masks = np.zeros((height, width), dtype=np.uint8)
        
            if result.masks is not None:
                for seg in result.masks.xyn:
                    seg[:, 0] *= width
                    seg[:, 1] *= height
                    segment = np.array(seg, dtype=np.int32)
                    segmentation_contours_idx.append(segment)
                    if len(segment) > 0 and len(segment[0]) > 0:
                        segments = segment.reshape((-1, 1, 2))
                        cv2.fillPoly(masks, [segments], 255)
        
            bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
            class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        
            if len(bboxes) > 0 and len(bboxes) == len(class_ids) == len(segmentation_contours_idx):
                for class_id, seg in zip(class_ids, segmentation_contours_idx):


                    if len(seg) == 0:
                        continue
                    if class_id == 0:  # boxer
                        if not is_valid_polygon(seg):
                            continue
                        dominant_color = get_dominant_hue(org_frame, seg)
                        if dominant_color == None:
                            continue
                        current_masks.append(dominant_color)
                    if class_id == 2:  # gloves
                        if not is_valid_polygon(seg):
                            continue
                        dominant_color = get_dominant_hue(org_frame, seg)
                        if dominant_color == None:
                            continue
                        gloves_color.append(dominant_color)
                        
                if len(current_masks) == 2: 
                    if current_masks[0] < current_masks[1]:
                        temp = current_masks[0]
                        current_masks[0] = current_masks[1]
                        current_masks[1] = temp
                else:
                    current_masks = None
                    
                if len(gloves_color) > 1 and len(gloves_color) < 5:
                    for i in range(len(gloves_color)):
                        for j in range(0, len(gloves_color) - i - 1):
                            if gloves_color[j] < gloves_color[j + 1]:
                                gloves_color[j], gloves_color[j + 1] = gloves_color[j + 1], gloves_color[j]
                else:
                    gloves_color = None
                
                b_gloves = []
                r_gloves = []
                
                for class_id, seg in zip(class_ids, segmentation_contours_idx):
                    if len(seg) == 0:
                        continue

                    
                    if current_masks == None or gloves_color == None:
                        continue
                    
                    red_color = [0,0,255]
                    blue_color = [255,0,0]
                    
                    if class_id == 1:  # head
                        if not is_valid_polygon(seg):
                            continue
                        dominant_color = get_dominant_hue(org_frame, seg)
                        if dominant_color == None:
                            continue
                        if dominant_color > 135 or dominant_color < 10:
                            cv2.polylines(frame, [seg], True, red_color, 2)
                            head_poly[0] = seg
                        else:
                            cv2.polylines(frame, [seg], True, blue_color, 2)
                            head_poly[1] = seg
                    elif class_id == 0:  # boxer
                        if not is_valid_polygon(seg):
                            continue
                        dominant_color = get_dominant_hue(org_frame, seg)
                        if dominant_color == None:
                            continue
                        if dominant_color > 135 or dominant_color < 5:
                            cv2.polylines(frame, [seg], True, red_color, 2)
                            boxer_poly[0] = seg
                        else:
                            cv2.polylines(frame, [seg], True, blue_color, 2)
                            boxer_poly[1] = seg
                    
                    if class_id == 2:  # gloves
                        if not is_valid_polygon(seg):
                            continue
                        dominant_color = get_dominant_hue(org_frame, seg)
                        if dominant_color == None:
                            continue
                        if dominant_color < 140 and dominant_color > 10:  # 藍色
                            cv2.polylines(frame, [seg], True, blue_color, 2)
                            b_gloves.append(dominant_color)
                            if gloves_poly[2] is None:
                                gloves_poly[2] = seg
                            else:
                                gloves_poly[3] = seg
                        else:
                            cv2.polylines(frame, [seg], True, red_color, 2)
                            r_gloves.append(dominant_color)
                            if gloves_poly[0] is None:
                                gloves_poly[0] = seg
                            else:
                                gloves_poly[1] = seg
                is_frame_put = 0
                if head_poly is not None and boxer_poly is not None and gloves_poly is not None and len(b_gloves) < 3 and len(r_gloves) < 3:
                    
                    try:
                        if (is_valid_polygon(gloves_poly[2]) or is_valid_polygon(gloves_poly[3])) and is_valid_polygon(head_poly[0]):
                            # 檢測藍色拳套擊中紅方頭部
                            if (Polygon(gloves_poly[3]).distance(Polygon(head_poly[0])) < 3 or 
                                Polygon(gloves_poly[2]).distance(Polygon(head_poly[0])) < 3 or 
                                Polygon(head_poly[0]).intersects(Polygon(gloves_poly[2])) or 
                                Polygon(head_poly[0]).intersects(Polygon(gloves_poly[3]))):
                                
                                try:
                                    cv2.polylines(frame, [gloves_poly[2].reshape((-1, 1, 2)).astype(np.int32)], True, [255,255,0], 2)
                                except Exception:
                                    pass
                                try:
                                    cv2.polylines(frame, [gloves_poly[3].reshape((-1, 1, 2)).astype(np.int32)], True, [255,255,0], 2)
                                except Exception:
                                    pass

                                sta_list.append("藍擊中紅_頭部")
                                is_frame_put = 1
                    except Exception:
                        print(Exception)
                    
                    try:
                        if (is_valid_polygon(gloves_poly[2]) or is_valid_polygon(gloves_poly[3])) and is_valid_polygon(boxer_poly[0]):
                            # 檢測藍色拳套擊中紅方身體
                            if (Polygon(gloves_poly[3]).distance(Polygon(boxer_poly[0])) < 3 or 
                                Polygon(gloves_poly[2]).distance(Polygon(boxer_poly[0])) < 3 or 
                                Polygon(boxer_poly[0]).intersects(Polygon(gloves_poly[2])) or 
                                Polygon(boxer_poly[0]).intersects(Polygon(gloves_poly[3]))):
                                
                                try:
                                    cv2.polylines(frame, [gloves_poly[2].reshape((-1, 1, 2)).astype(np.int32)], True, [255,255,0], 2)
                                except Exception:
                                    pass
                                try:
                                    cv2.polylines(frame, [gloves_poly[3].reshape((-1, 1, 2)).astype(np.int32)], True, [255,255,0], 2)
                                except Exception:
                                    pass
                                
                                sta_list.append("藍擊中紅_身體")
                                is_frame_put = 1
                    except Exception:
                        print(Exception)
                    try:
                        if (is_valid_polygon(gloves_poly[1]) or is_valid_polygon(gloves_poly[0])) and is_valid_polygon(head_poly[1]):
                            # 檢測紅色拳套擊中藍方頭部
                            if (Polygon(gloves_poly[1]).distance(Polygon(head_poly[1])) < 3 or 
                                Polygon(gloves_poly[0]).distance(Polygon(head_poly[1])) < 3 or 
                                Polygon(head_poly[1]).intersects(Polygon(gloves_poly[0])) or 
                                Polygon(head_poly[1]).intersects(Polygon(gloves_poly[1]))):
                                
                                try:
                                    cv2.polylines(frame, [gloves_poly[0].reshape((-1, 1, 2)).astype(np.int32)], True, (0,255,255), 2)
                                except Exception:
                                    pass
                                try:
                                    cv2.polylines(frame, [gloves_poly[1].reshape((-1, 1, 2)).astype(np.int32)], True, (0,255,255), 2)
                                except Exception:
                                    pass     
                                
                                sta_list.append("紅擊中藍_頭部")
                                is_frame_put = 1
                    except Exception:
                        print(Exception)

                    try:
                        if (is_valid_polygon(gloves_poly[1]) or is_valid_polygon(gloves_poly[0])) and is_valid_polygon(boxer_poly[1]):
                            # 檢測紅色拳套擊中藍方身體
                            if (Polygon(gloves_poly[1]).distance(Polygon(boxer_poly[1])) < 3 or 
                                Polygon(gloves_poly[0]).distance(Polygon(boxer_poly[1])) < 3 or 
                                Polygon(boxer_poly[1]).intersects(Polygon(gloves_poly[0])) or 
                                Polygon(boxer_poly[1]).intersects(Polygon(gloves_poly[1]))):
                                
                                try:
                                    cv2.polylines(frame, [gloves_poly[0].reshape((-1, 1, 2)).astype(np.int32)], True, (0,255,255), 2)
                                except Exception:
                                    pass
                                try:
                                    cv2.polylines(frame, [gloves_poly[1].reshape((-1, 1, 2)).astype(np.int32)], True, (0,255,255), 2)
                                except Exception:
                                    pass
                                
                                sta_list.append("紅擊中藍_身體")
                                is_frame_put = 1
                    except Exception:
                        pass

                    if is_frame_put == 0:
                        sta_list.append("未擊中")
                        

            if len(sta_list) == 0:
                sta_list.append("資料不足")
            
            queue.put(sta_list)

            encoded_image = cv2.imencode('.jpg', frame)[1]
            base64_str = base64.b64encode(encoded_image)
            image_queue.put(base64_str)

            #這是直播
            push_queue.put(encoded_image)   #這是推流queue
            
        
            frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
            cv2.imshow(f"Camera {process_id}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        sys.exit(0)
    finally:
        pass

# 計分整合
def mix_score(cam1_queue, cam2_queue, cam3_queue, image1_queue, image2_queue, image3_queue, barrier, shared, game_id):
    print("Mix score process waiting at barrier")
    rhb_sta = 0
    rhh_sta = 0
    bhb_sta = 0
    bhh_sta = 0
    counter = 0
    second = 0
    
    
    try:
        barrier.wait()
    except Exception as e:
        print(f"Barrier error: {e}")
    print("Mix score process starting")
    no_hit =10
    while True:
        try:
            if not cam1_queue.empty() and  not cam2_queue.empty() and not cam3_queue.empty():
                frame_results = []
                
                frame_results.extend(cam1_queue.get(timeout=0.1))
                frame_results.extend(cam2_queue.get(timeout=0.1))
                frame_results.extend(cam3_queue.get(timeout=0.1))
                

                print(frame_results)
                if rhb_sta > 0: rhb_sta -= 1
                if rhh_sta > 0: rhh_sta -= 1
                if bhb_sta > 0: bhb_sta -= 1
                if bhh_sta > 0: bhh_sta -= 1

                counter += 1
                if counter == 15:
                    counter = 0
                    second += 1
                
                img_list = []
                img1 = image1_queue.get().decode('utf-8')
                img2 = image2_queue.get().decode('utf-8')
                img3 = image3_queue.get().decode('utf-8')
                img_list.extend([img1, img2, img3])
                data = {
                    "type": "uploadImages",
                    "gameId": game_id,
                    "files": img_list,
                    "score":""
                }
                
                miss_count = frame_results.count("未擊中")
                #insufficient_count = frame_results.count("資料不足")
                rhitb_body = frame_results.count("紅擊中藍_身體")
                rhitb_head = frame_results.count("紅擊中藍_頭部")
                bhitr_body = frame_results.count("藍擊中紅_身體")
                bhitr_head = frame_results.count("藍擊中紅_頭部")
                is_no_hit = 1
                
                if miss_count >= abs(2) and miss_count != 0:
                    continue


                if rhitb_head >= abs(2) and rhh_sta == 0:
                    rhh_sta = no_hit
                    is_no_hit = 0
                    with shared.player1_headhit.get_lock():
                        shared.player1_headhit.value += 1
                    data["score"] = "紅方擊中頭部"
                    print("紅方擊中頭部")
                    scorer = "player1"
                    point = "head"
                elif rhitb_body >= abs(2) and rhb_sta == 0:
                    rhb_sta = no_hit
                    is_no_hit = 0
                    with shared.player1_bodyhit.get_lock():
                        shared.player1_bodyhit.value += 1
                    data["score"] = "紅方擊中身體"
                    print("紅方擊中身體")
                    scorer = "player1"
                    point = "body"

                
                if bhitr_head >= abs(2) and bhh_sta == 0:
                    bhh_sta = no_hit
                    is_no_hit = 0
                    with shared.player2_headhit.get_lock():
                        shared.player2_headhit.value += 1
                    if data["score"] == "":
                        data["score"] = "藍方擊中頭部"
                    elif data["score"] != "":
                        data["score"] += ",藍方擊中頭部"
                    print("藍方擊中頭部")
                    scorer = "player2"
                    point = "head"
                elif bhitr_body >= abs(2) and bhb_sta == 0:
                    bhb_sta = no_hit
                    is_no_hit = 0
                    with shared.player2_bodyhit.get_lock():
                        shared.player2_bodyhit.value += 1
                    print("藍方擊中身體")
                    if data["score"] == "":
                        data["score"] = "藍方擊中身體"
                    elif data["score"] != "":
                        data["score"] += " 藍方擊中身體"
                    scorer = "player2"
                    point = "body"
                
                if is_no_hit:
                    continue


                print(data["score"])
                img_info = json.dumps(data)
                #這是Websocket
                asyncio.run(send_image(img_info)) 
                #這是DB更新總分
                update_db_total(shared, game_id)


                #insert_db_event(scorer, point, second, game_id)
            else:
                pass    
        except Empty:
            continue
        except KeyboardInterrupt:
            os._exit(0)
        except Exception as e:
            print(f"Error in mix_score: {e}")
            continue

# 資料庫更新
def update_db_total(shared, game_id):
    db_settings = {
        "host": f"{ip_addr}",
        "port": 3306,
        "user": "cvboxingReferee",
        "password": "your_password",
        "db": "cvboxingReferee",
        "charset": "utf8"
    }
    
    with shared.player1_bodyhit.get_lock(), \
         shared.player2_bodyhit.get_lock(), \
         shared.player1_headhit.get_lock(), \
         shared.player2_headhit.get_lock():
        
        p1_body = shared.player1_bodyhit.value
        p2_body = shared.player2_bodyhit.value
        p1_head = shared.player1_headhit.value
        p2_head = shared.player2_headhit.value
    
    conn = pymysql.connect(**db_settings)
    with conn.cursor() as cursor:
        command = "UPDATE game_list SET p1_head = %s, p2_head = %s, p1_body = %s, p2_body = %s WHERE game_id = %s"
        cursor.execute(command, (p1_head, p2_head, p1_body, p2_body, game_id))
        conn.commit()
        print("db updated")
    conn.close()

# 資料庫插入
def insert_db_event(scorer, point, time, game_id):
    db_settings = {
        "host": f"{ip_addr}",
        "port": 3306,
        "user": "cvboxingReferee",
        "password": "your_password",
        "db": "cvboxingReferee",
        "charset": "utf8"
    }
    
    conn = pymysql.connect(**db_settings)
    with conn.cursor() as cursor:
        command = "INSERT INTO game_info(list_id, scorer, point, time)VALUES(%s, %s, %s, %s)"
        cursor.execute(command, (game_id, scorer, point, time))
        conn.commit()
        print("db inserted")
    conn.close()
