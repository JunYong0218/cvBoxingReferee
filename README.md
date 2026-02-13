# 基於電腦視覺的拳擊競賽即時判分輔助系統

本系統使用 YOLOv8 物件偵測技術，結合多攝影機視角整合，實現拳擊比賽的自動計分與即時直播功能。
(此為重新模組化拆分後的版本，僅包含影像處理、判定邏輯、推流部分，不包含前端網頁部分)

## 系統功能

- YOLOv8 即時物件偵測與追蹤
- 三路攝影機多視角整合驗證
- HSV 色相分析自動識別選手
- 頭部與身體擊中分別計分
- RTMP 多路同步直播推流
- WebSocket 即時分數傳輸
- MySQL 比賽數據完整記錄

## 檔案結構

```
config.py       - 系統配置、全域變數、共享類別
core_worker.py  - YOLO 模型初始化
proces.py       - AI 辨識、計分邏輯、資料庫操作
streamer.py     - RTMP 推流處理
main.py         - 主程式、進程管理
```

## 系統架構

系統採用多進程並行架構：

- 3 個攝影機讀取進程 (get_frame)
- 3 個 AI 辨識進程 (cam)
- 1 個計分整合進程 (mix_score)
- 3 個推流線程 (push_stream)

各進程透過 Queue 進行資料傳遞，使用 Barrier 同步啟動，SharedVars 共享計分數據。

## 環境需求

### 硬體

- NVIDIA GPU (支援 CUDA)
- 3 台 USB 攝影機或 RTSP 網路攝影機
- RTMP 伺服器
- MySQL 資料庫伺服器

### 軟體

Python 3.8 或更高版本，相依套件如下：

```bash
pip install -r requirements.txt
```

主要套件：
- opencv-python (影像處理)
- ultralytics (YOLOv8)
- numpy (數值運算)
- shapely (幾何運算)
- websockets (即時通訊)
- pymysql (資料庫連線)

## 判分機制

### 多視角驗證

三個攝影機獨立進行辨識，至少兩個攝影機判定為擊中才確認得分，提高判定準確度。

### 選手識別

透過 HSV 色相空間分析護具顏色：
- 紅方：色相值 > 135 或 < 10
- 藍方：色相值介於 10 至 135 之間

### 擊中判定

計算拳套與對手身體的多邊形距離，當距離小於 3 像素或多邊形相交時判定為擊中。系統設有 10 幀冷卻時間避免重複計分。

### 計分規則

系統記錄擊中次數，並儲存至資料庫：
- 頭部擊中次數 (p1_head, p2_head)
- 身體擊中次數 (p1_body, p2_body)

實際分數計算由後端或評分系統根據擊中次數決定。

## 系統配置

### 攝影機配置

預設使用 USB 攝影機（裝置編號 0, 1, 2）。若使用 RTSP 網路攝影機，修改 `main.py` 中的攝影機參數：

```python
p6 = Process(target=get_frame, args=(
    "rtsp://192.168.0.100:554/stream",  # RTSP 位址
    video_queue1, barrier, 3, shared
))
```

### 運算資源配置

預設使用 GPU 進行推論。若需改用 CPU，修改 `core_worker.py`：

```python
def init_model():
    model = YOLO('new_10.pt')
    model.to('cpu')
    return model
```

## 注意事項

1. YOLO 模型檔案 `new_10.pt` 無提供，需使用標註過的拳擊數據集訓練。

2. 資料庫密碼儲存於 `proces.py` 的 `db_settings` 中，上傳前請確認已改為 `your_password` 範例值。

3. 系統需要 RTMP 伺服器支援，推流位址格式為 `rtmp://[IP]/hls/[game_id]-[cam_id]`。

4. 確保 FFmpeg 已正確安裝並加入系統 PATH。

## 開發團隊

國立聯合大學資訊工程系

### 團隊分工 (Team Contributions)

#### **蕭鈞永**
* **核心貢獻**：
    * 開發後端程式碼，包含影像讀取&處理、AI辨識、計分邏輯
    * 重新模組化程式碼、文檔撰寫
    * 進行訓練集數據標註

    

#### **Terry**
* **核心貢獻**：
    * 開發後端程式碼，包含多進程管理、RTMP 推流、WebSocket 即時傳輸
    * 微調 YOLOv8 模型
    * 進行訓練集數據標註

#### **linyushi**
* **核心貢獻**：
    * 撰寫自動化工具腳本
    * 進行訓練集數據標註

#### **MingYao**
* **核心貢獻**：
    * 實作前端網頁
    * 串接後端推流、資料庫處理
