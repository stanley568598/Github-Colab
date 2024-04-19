# Mask Detection (YOLO)

## 介紹

這是一個簡單的「物件偵測」練習，使用「YOLO」偵測圖片中「戴著口罩的人臉」、「沒戴著口罩的人臉」、「沒戴好口罩的人臉」。

> <details>
> 
> <summary>更多相關內容介紹</summary>
> 
> <br>
> 
> > ### 物件偵測 與 影像辨識
> >
> > 物件偵測，除了要判斷影像中的所有物體各自屬於哪個類別之外，還要找出物體的位置。
> >
> > <details>
> >
> > <summary>更多詳細介紹</summary>
> > 
> > - CNN 對於物體的分類又快又好，但用 CNN 掃描，來辨識圖片中的各種物體的效率十分低下。
> > 
> >     - 最簡單的作法就是用 Sliding Windows 的概念，也就是用一個固定大小的框框，逐一的掃過整張圖片，每次框出來的圖像丟到 CNN 中去判斷類別。
> > 
> >     - 由於物體的大小是不可預知的，所以還要用不同大小的框框去偵測。
> >     
> >     - 表示我們對單一影像需要掃描非常多次，每掃一次都需要算一次 CNN，這會耗費大量的運算資源，而且速度很慢。
> > 
> > - YOLO 是一種用來辨識影像中物體的 AI 模型，讓電腦能夠快速地識別出一張圖片中的物體 + 它們的位置的技術。
> > 
> >     - 原本的物件偵測任務是利用「分類器」來進行，但 YOLO 將物件偵測視為一個「回歸 ( regression )」任務。
> > 
> >     - 從空間中分割出邊界框 ( Bounding Box ) 並且計算出類別「機率」。
> > 
> >     - 輸出包含 bounding box、confidence 及 class probability。
> > 
> > </details>
> 
> > ### YOLO
> > 
> > YOLO 的全名是 “You Only Look Once”，意思是電腦只需要看一眼圖片，就能完成物件的影像辨識和定位。
> > 
> > <details>
> >
> > <summary>更多詳細介紹</summary>
> >
> > - YOLO 物件偵測方式：
> > 
> >     將原圖拆成很多個 grid cell，然後在「每個 grid cell」進行「兩個 bounding box 的預測」和「屬於哪個類別的機率預測」，最後用閾值和 NMS ( Non-Maximum Suppression，非極大值抑制 ) 的方式得到結果。
> > 
> >     1. YOLO 會把原圖先平均分成 S×S 格。
> > 
> >         - 這邊假設原圖大小為 100 x 100，S 為 5，原圖就會被平均分成 5 × 5 的 grid cell ( 大小 = 20 × 20 )。
> > 
> >     2. 每個 grid cell 必須要負責預測「B 個 bounding boxes」和「屬於每個類別的機率」，每個 bounding box 會帶有 5 個預設值 ( x, y, w, h, and confidence )。
> > 
> >         - ( x, y ) 用來表示某一個物件在這個 grid cell 的中心座標，這個物件相對應的寬高分別為 w, h。
> >         
> >         - confidence 則是用來表示這個物件是否為一個物件的信心程度 ( confidence score )。
> > 
> >         - 參考資料：https://developer.aliyun.com/article/1309596
> > 
> >     3. 整體的概念就是如果要「被偵測的物件中心」落在哪一個 grid cell，那個 grid cell 就要負責偵測這個物件。
> >         
> >         - 如果物件在 grid cell 內，confidence score 就等於「預測的 bounding box」和「ground truth 事實」的 IOU ( intersection over union，重疊性 )。反之，如果在某個 grid cell 沒有任何物件，這時候 confidence score 就會是 0，
> > 
> > - YOLO 僅利用一個神經網路，進行一次 CNN 計算，來直接預測「邊界框」及「類別機率」。
> >     
> >     - 因為整個偵測過程只有使用單一個神經網路，因此可以視為是一個 End-to-End 的優化過程。
> > 
> >     - 這樣統一的架構，將物件定位和分類一起完成，執行速度十分快速，效率極高。
> > 
> > - YOLO 在邊界框預測上有很強的「空間限制」：每一個網格 ( grid ) 僅能生成預測「兩個」邊界框 ( Bounding Box )，並且只能有一個目標類別 ( class )。
> > 
> >     - 限制了 YOLO 對相鄰目標的預測。換句話說，YOLO 對於成群聚集 ( 密集 ) 的小物件 ( 例如鳥群、人群 ) 有預測上的困難。
> >     
> >     - YOLO 在精度上的缺失：雖然它可以快速識別影像中的目標，但很難精確定位某些目標，尤其是小物件。
> > 
> > - YOLO 是基於全域圖片進行推理，不像滑窗和 region proposal-base 演算法那樣只是基於感興趣區域做推理。
> > 
> >     - 由於 yolo 訓練和推理都是基於整張圖片，而 Fast R-CNN 是基於局部感興趣區域訓練，所以 Fast R-CNN 將背景誤認為目標的錯誤較多，yolo 的背景誤報相對地少了一半。
> > 
> >     - YOLO 對全域資訊有較好的效果 ( 大物件 / 離影像邊界很近的物件 )，但在小範圍 ( 小 / 密集 ) 的資訊上表現較差。
> > 
> > </details>
> 
> </details>

## 功能

- [x] 使用 GPU 計算。
- [x] 建置 Darknet 執行檔：YOLO 物件偵測技術的運算模組。
- [x] 讀取影像 ( .jpg / .jpeg )、標記檔 ( .xml )。
- [x] 建立 YOLO 資料格式：綁定圖像與標記範圍。
- [x] 建立參數檔 ( .cfg )、訓練結果的權重資料 ( .weights )。
- [x] 使用 darknet 偵測人臉、輸出戴口罩辨識結果。

## 環境部署

開發環境：Google Collaboratory

> Original file is located at：[Mask Detection (YOLO)](https://colab.research.google.com/drive/1Uz4uWqZA_iX05JBIcrV3KOCohoY_rfE2)

### 注意事項

1. 此專案部署時，Colab 環境會開啟一台虛擬機，並且使用 google 雲端硬碟來存取檔案。

    - 因此，執行後 Colab 將要求授權，請求連結到 Google Drive 的權限。
    
    - 請進入「執行階段 >> 變更執行階段類型」，啟用 GPU 為硬體加速器，使得 darknet 能正常運行。

2. 完成 Colab 環境部署後，在 Google Drive 將出現一個 Colab Notebooks 的資料夾。

    - 請將本專案的「AIDataset」上傳至 Colab Notebooks 的資料夾之中，以利 Colab 透過 google drive 取得訓練資料。

## 使用套件

- darknet：深度學習框架，https://github.com/pjreddie/darknet.git 。
- YOLO v3 技術：引入一個具有多尺度預測的新架構，使模型能夠更準確地檢測不同大小的物件。

### 踩坑紀錄

1. darknet 編譯配置：Makefile。
    
    Issue：下載完 darknet 原始碼後需要使用 make 指令進行編譯，編譯的過程實際上是執行「Makefile」中的語句。要成功編譯，還需根據我們自身伺服器環境的情況對 MakeFile 文件進行修改。
    
    <details>
    
    <summary>查看解決方案</summary>
    
    - Solve：修改 GPU, CUDNN, OpenCV 設定為 1，啟動調用。

        ```py
        ! sed -i "s/GPU=0/GPU=1/g" darknet/Makefile
        ! sed -i "s/CUDNN=0/CUDNN=1/g" darknet/Makefile
        ! sed -i "s/OPENCV=0/OPENCV=1/g" darknet/Makefile
        ```

    </details>

2. 找不到 opencv。

    Issue：此錯誤訊息發生於【開始編譯 darknet】( ! cd darknet; make )

    ```
    Package opencv was not found in the pkg-config search path.
    Perhaps you should add the directory containing `opencv.pc'
    to the PKG_CONFIG_PATH environment variable
    No package 'opencv' found
    ```
    
    <details>
    
    <summary>查看解決方案</summary>
    
    - Solve：修改 Makefile，將 opencv 改成 opencv4。

        ```py
        makefiletemp = open('darknet/Makefile','r+')
        list_of_lines = makefiletemp.readlines()

        list_of_lines[44] = "LDFLAGS+= `pkg-config --libs opencv4` -lstdc++" + "\n"
        list_of_lines[45] = "COMMON+= `pkg-config --cflags opencv4`" + "\n"

        makefiletemp = open('darknet/Makefile','w')
        makefiletemp.writelines(list_of_lines)
        makefiletemp.close()
        ```
    
    </details>

3. 最新版 CUDA，已棄用 cudaThreadSynchronize 函數。

    Issue：此錯誤訊息發生於【開始編譯 darknet】( ! cd darknet; make )

    ```
    ./src/gemm.c: In function ‘time_gpu’:
    ./src/gemm.c:232:9: warning: ‘cudaThreadSynchronize’ is deprecated [-Wdeprecated-declarations]
        232 |         cudaThreadSynchronize();
            |         ^~~~~~~~~~~~~~~~~~~~~
    In file included from /usr/local/cuda/include/cuda_runtime.h:95,
                        from include/darknet.h:11,
                        from ./src/utils.h:5,
                        from ./src/gemm.c:2:
    /usr/local/cuda/include/cuda_runtime_api.h:1069:57: note: declared here
        1069 | extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaThreadSynchronize(void);
             |                                                         ^~~~~~~~~~~~~~~~~~~~~
    ```

    <details>
    
    <summary>查看解決方案</summary>

    - Solve：修改「darknet/src/gemm.c」的程式碼。

        - cuda 在 10.0 及之後的版本中刪除了 cudaThreadSynchronize 函數，改成使用另一個函數 cudaDeviceSynchronize。

        - 觀察報錯可知 error 出現在 gemm.c 的 232 行。

        ```py
        temp = open('darknet/src/gemm.c','r+')
        lines = temp.readlines()

        lines[231] = lines[231].replace('cudaThreadSynchronize', 'cudaDeviceSynchronize')

        temp = open('darknet/src/gemm.c','w')
        temp.writelines(lines)
        temp.close()
        ```

    </details>

4. 最新版 cuDNN，已修改大量舊版寫法。

    Issue：此錯誤訊息發生於【開始編譯 darknet】( ! cd darknet; make )

    ```
    gcc -Iinclude/ -Isrc/ -DOPENCV `pkg-config --cflags opencv4` -DGPU -I/usr/local/cuda/include/ -DCUDNN  -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -Ofast -DOPENCV -DGPU -DCUDNN -c ./src/convolutional_layer.c -o obj/convolutional_layer.o
    ./src/convolutional_layer.c: In function ‘cudnn_convolutional_setup’:
    ./src/convolutional_layer.c:148:5: warning: implicit declaration of function ‘cudnnGetConvolutionForwardAlgorithm’; did you mean ‘cudnnGetConvolutionForwardAlgorithm_v7’? [-Wimplicit-function-declaration]
        148 |     cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            |     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            |     cudnnGetConvolutionForwardAlgorithm_v7
    ./src/convolutional_layer.c:153:13: error: ‘CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT’ undeclared (first use in this function)
        153 |             CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
            |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    compilation terminated due to -Wfatal-errors.
    make: *** [Makefile:89: obj/convolutional_layer.o] Error 1
    ```
    
    <details>
    
    <summary>查看解決方案</summary>

    - Solve：修改「darknet/src/convolutional_layer.c」的程式碼。

        - cudnn 在 8.x 及之後的版本，已修改大量 cudnn7.x 的寫法。
    
        - 例如：對於原生於 cudnn7.x 的 darknet，新版本 cudnn8.x 已移除 CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT 相關定義。

        - 修改 convolutional_layer.c，增加針對 CUDNN_MAJOR>=8 的處理。

        ```py
        temp = open('darknet/src/convolutional_layer.c','r+')
        lines = temp.readlines()

        new_code =  """
            #if CUDNN_MAJOR >= 8
            int returnedAlgoCount;
            cudnnConvolutionFwdAlgoPerf_t		fw_results[ 2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT ];
            cudnnConvolutionBwdDataAlgoPerf_t	bd_results[ 2 * CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT ];
            cudnnConvolutionBwdFilterAlgoPerf_t	bf_results[ 2 * CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT ];

            cudnnFindConvolutionForwardAlgorithm(cudnn_handle(),
                    l->srcTensorDesc,
                    l->weightDesc,
                    l->convDesc,
                    l->dstTensorDesc,
                    CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
                    &returnedAlgoCount,
                fw_results);

            for(int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex){
                #if PRINT_CUDNN_ALGO > 0
                printf("^^^^ %s for Algo %d: %f time requiring %llu memory\\n",
                    cudnnGetErrorString(fw_results[algoIndex].status),
                    fw_results[algoIndex].algo, fw_results[algoIndex].time,
                    (unsigned long long)fw_results[algoIndex].memory
                );
                #endif
                if( fw_results[algoIndex].memory < MEMORY_LIMIT){
                    l->fw_algo = fw_results[algoIndex].algo;
                    break;
                }
            }

            cudnnFindConvolutionBackwardDataAlgorithm(cudnn_handle(),
                    l->weightDesc,
                    l->ddstTensorDesc,
                    l->convDesc,
                    l->dsrcTensorDesc,
                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT,
                    &returnedAlgoCount,
                bd_results);

            for(int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex){
                #if PRINT_CUDNN_ALGO > 0
                printf("^^^^ %s for Algo %d: %f time requiring %llu memory\\n",
                    cudnnGetErrorString(bd_results[algoIndex].status),
                    bd_results[algoIndex].algo, bd_results[algoIndex].time,
                    (unsigned long long)bd_results[algoIndex].memory
                );
                #endif
                if( bd_results[algoIndex].memory < MEMORY_LIMIT){
                    l->bd_algo = bd_results[algoIndex].algo;
                    break;
                }
            }

            cudnnFindConvolutionBackwardFilterAlgorithm(cudnn_handle(),
                    l->srcTensorDesc,
                    l->ddstTensorDesc,
                    l->convDesc,
                    l->dweightDesc,
                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT,
                    &returnedAlgoCount,
                bf_results);

            for(int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex){
                #if PRINT_CUDNN_ALGO > 0
                printf("^^^^ %s for Algo %d: %f time requiring %llu memory\\n",
                    cudnnGetErrorString(bf_results[algoIndex].status),
                    bf_results[algoIndex].algo, bf_results[algoIndex].time,
                    (unsigned long long)bf_results[algoIndex].memory
                );
                #endif
                if( bf_results[algoIndex].memory < MEMORY_LIMIT){
                    l->bf_algo = bf_results[algoIndex].algo;
                    break;
                }
            }
            #else \n"""

        lines.insert(171, "\n   #endif \n\n")

        lines.insert(146, new_code)
        
        define_code = """
        #define PRINT_CUDNN_ALGO 0
        #define MEMORY_LIMIT 2000000000
        """

        lines.insert(10, define_code)

        temp = open('darknet/src/convolutional_layer.c','w')
        temp.writelines(lines)
        temp.close()
        ```

    </details>

5. 最新版 opencv4 的補丁，已修改舊版 opencv 所使用的資料格式 ( IplImage ) 與 前綴詞 ( CV_ )。

    Issue：此錯誤訊息發生於【開始編譯 darknet】( ! cd darknet; make )

    ```
    g++ -Iinclude/ -Isrc/ -DOPENCV `pkg-config --cflags opencv4` -DGPU -I/usr/local/cuda/include/ -DCUDNN  -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -Ofast -DOPENCV -DGPU -DCUDNN -c ./src/image_opencv.cpp -o obj/image_opencv.o
    ./src/image_opencv.cpp:12:1: error: ‘IplImage’ does not name a type
         12 | IplImage *image_to_ipl(image im)
            | ^~~~~~~~
    compilation terminated due to -Wfatal-errors.
    make: *** [Makefile:86: obj/image_opencv.o] Error 1
    ```

    ```
    g++ -Iinclude/ -Isrc/ -DOPENCV `pkg-config --cflags opencv4` -DGPU -I/usr/local/cuda/include/ -DCUDNN  -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -Ofast -DOPENCV -DGPU -DCUDNN -c ./src/image_opencv.cpp -o obj/image_opencv.o
    ./src/image_opencv.cpp: In function ‘void* open_video_stream(const char*, int, int, int, int)’:
    ./src/image_opencv.cpp:122:20: error: ‘CV_CAP_PROP_FRAME_WIDTH’ was not declared in this scope
        122 |     if(w) cap->set(CV_CAP_PROP_FRAME_WIDTH, w);
            |                    ^~~~~~~~~~~~~~~~~~~~~~~
    compilation terminated due to -Wfatal-errors.
    make: *** [Makefile:86: obj/image_opencv.o] Error 1
    ```

    <details>
    
    <summary>查看解決方案</summary>

    - Solve：修改「darknet/src/image_opencv.cpp」的程式碼。

        - 刪除 IplImage 轉換函數，直接將影像與 Mat 相互轉換。
        
        - Capture ( 擷取 ) 屬性，不再以 CV_ 開頭，因此需要從所有屬性 ( properties ) 中刪除該相關前綴，都以 CAP_PROP_ 開頭。

        ```py
        temp = open('darknet/src/image_opencv.cpp','r+')
        lines = temp.readlines()

        # 注意：須避免下方 modify_code 的 CV_8UC 被置換掉。
        for i in range(len(lines)):
            lines[i] = lines[i].replace("CV_","")

        lines[10] = "/* \n"
        lines[67] = "*/ \n"

        modify_code = """

        Mat image_to_mat(image im)
        {
            image copy = copy_image(im);
            constrain_image(copy);
            if(im.c == 3) rgbgr_image(copy);

            Mat m(cv::Size(im.w,im.h), CV_8UC(im.c));
            int x,y,c;

            int step = m.step;
            for(y = 0; y < im.h; ++y){
                for(x = 0; x < im.w; ++x){
                    for(c= 0; c < im.c; ++c){
                        float val = im.data[c*im.h*im.w + y*im.w + x];
                        m.data[y*step + x*im.c + c] = (unsigned char)(val*255);
                    }
                }
            }

            free_image(copy);
            return m;
        }

        image mat_to_image(Mat m)
        {

            int h = m.rows;
            int w = m.cols;
            int c = m.channels();
            image im = make_image(w, h, c);
            unsigned char *data = (unsigned char *)m.data;
            int step = m.step;
            int i, j, k;

            for(i = 0; i < h; ++i){
                for(k= 0; k < c; ++k){
                    for(j = 0; j < w; ++j){
                        im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
                    }
                }
            }
            rgbgr_image(im);
            return im;
        }
        \n"""

        lines.insert(68, modify_code)

        temp = open('darknet/src/image_opencv.cpp','w')
        temp.writelines(lines)
        temp.close()
        ```

    </details>

6. 新版本 CUDA，已棄用過時的 GPU 架構 'compute_30'。

    Issue：此錯誤訊息發生於【開始編譯 darknet】( ! cd darknet; make )

    ```
    nvcc fatal   : Unsupported gpu architecture 'compute_30'
    make: *** [Makefile:92: obj/convolutional_kernels.o] Error 1
    ```

    <details>
    
    <summary>查看解決方案</summary>

    - Solve：修改 Makefile。
        
        - 註解 【ARCH= -gencode arch=compute_30,code=sm_30 \】。
        
        - 下一行順延補上 【ARCH= 】。

        ```py
        makefiletemp = open('darknet/Makefile','r+')
        list_of_lines = makefiletemp.readlines()

        list_of_lines[6] = "# " + list_of_lines[6]
        list_of_lines[7] = "ARCH= " + list_of_lines[7].strip() + "\n"

        makefiletemp = open('darknet/Makefile','w')
        makefiletemp.writelines(list_of_lines)
        makefiletemp.close()
        ```

    </details>

7. YOLO v3 訓練參數 ( .cfg ) 預設為 80 個類別。
    
    Issue：查看訓練參數 ( ! sed -n -e 127p -e 135p -e 171p -e 177p /content/cfg_mask/yolov3-tiny.cfg )
    
    ```
    filters=255
    classes=80
    filters=255
    classes=80
    ```
    
    <details>
    
    <summary>查看解決方案</summary>

    - Solve：修改「yolov3-tiny.cfg」。

        - YOLOV3 偵測的濾鏡【 filter = ( C + 5 ) * B 】。
        
            > C 是 class 類別數量；B 是每個 Feature Map 可以偵測的 Bounding Box 數量；
            > 5 代表的是此 Bounding Box 的網格特徵 ( x, y, w, h, confidence score )。

        - 原本設定 80 個 class：filter = ( 80 + 5 ) * 3 = 255。
        
        - 調整成，3 個 class：filter = ( 3 + 5 ) * 3 = 24。
            
            > good：有戴口罩；bad：沒戴口罩；none：沒戴好口罩
    
        ```py
        # line 127: filters
        ! sed -i '127s/255/24/' /content/cfg_mask/yolov3-tiny.cfg

        # line 135: classes
        ! sed -i '135s/80/3/' /content/cfg_mask/yolov3-tiny.cfg

        # line 171: filters
        ! sed -i '171s/255/24/' /content/cfg_mask/yolov3-tiny.cfg

        # line 177: classes
        ! sed -i '177s/80/3/' /content/cfg_mask/yolov3-tiny.cfg
        ```

    </details>

## Demo

> 完整執行結果顯示於 (.ipynb) 檔：[YOLO_(Mask_Detection).ipynb](YOLO_(Mask_Detection).ipynb)

1. Input training data：輸入訓練影像 ( image ) 與 標籤 ( label )。

    - Input image
    
        ![input_image](./assets/images/1.%20input_image.JPG)
    
    - Input label
    
        ![input_label](./assets/images/1.%20input_label.JPG)

2. YOLO setting：設定 YOLO 參數。

    - YOLO introduction

        ![YOLO_intro](./assets/images/2-1.%20YOLO_intro.JPG)
    
    - YOLO method
        
        ![YOLO_method](./assets/images/2-2.%20YOLO_method.JPG)
        
        ![YOLO_method_illustration](./assets/images/2-3.%20YOLO_method_illustration.JPG)
    
    - YOLO parameters
    
        ![YOLO_parameters](./assets/images/2-4.%20YOLO_parameters.JPG)

3. Predictions：使用 YOLO 模型，辨識未知影像。

    ![predict_image](./assets/images/3.%20predict_image.JPG)

4. Predictions for different training epoch：比較不同訓練量的預測效果。

    - epoch = 40000
        
        ![predict_weights_from_40000_epoch](./assets/images/4.%20predict_weights_from_40000_epoch.JPG)

    - epoch = 70000
        
        ![predict_weights_from_70000_epoch](./assets/images/4.%20predict_weights_from_70000_epoch.JPG)

    - epoch = 130000
        
        ![predict_weights_from_130000_epoch](./assets/images/4.%20predict_weights_from_130000_epoch.JPG)

