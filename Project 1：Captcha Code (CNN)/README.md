# Captcha Code (CNN)

## 介紹

這是一個簡單的「圖像辨識」練習，使用「卷積神經網絡 ( CNN ) 」，進行「驗證碼 ( CAPTCHA )」的圖像辨識。

> <details>
> 
> <summary>更多相關內容介紹</summary>
> 
> > ## 卷積神經網絡 ( Convolutional Neural Network，CNN )
> > 
> > 神經網路是使用神經元形成節點網路的模型，透過每個神經元的計算，分析各種因素與結果的關聯性，找出權重大的成因，產生分類結果。
> > 
> > 不同於一般的神經網路只是單純的提取資料進行運算，「卷積神經網絡」增加了「卷積層 ( Convolution Layer )」及「池化層 ( Pooling Layer )」，讓 CNN 擁有能夠「看」到圖像或語音細節的能力。
> > <details>
> > 
> > <summary>更多詳細內容</summary>
> >
> > > ### Convolution Layer ( 卷積層 )
> > > 
> > > 卷積，有點像是電腦的眼睛，是 CNN 中最重要的工具，其用途是提取特徵，主要由兩個步驟組成的運算：滑動 + 內積。
> > > 
> > > <details>
> > > 
> > > <summary>更多詳細內容</summary>
> > > 
> > > - 人主要依靠局部特徵的不同來分辨事物。要讓電腦學習這些局部特徵，我們利用一個小框框 ( filter ) 來掃描圖片。當掃描到重要的特徵時，數值就會變大，不重要的數值就會很小。掃描後的數值又稱為 feature map。
> > > 
> > >     - 卷積核 ( Kernel )，又稱為 Filters, Features Detectors，主要目的是萃取出圖片當中的一些特徵 ( 例如：形狀、邊界 )。
> > > 
> > >     - 透過卷積核 ( Kernels ) 滑動對圖像做訊息提取，並藉由步長 ( Strides ) 與填充 ( Padding ) 控制圖像的長寬。
> > > 
> > >     - 利用 Filter ( kernel map ) 在輸入圖片上滑動並且持續進行矩陣內積後，卷積完得到的圖片稱之為 feature map。
> > > 
> > > - 根據每次卷積的值和位置，製作一個新的二維矩陣。它可以告訴我們在原圖的哪些地方可以找到該特徵。值越接近 1 的局部和該特徵越相符，值越接近 -1 則相差越大，至於接近值接近 0 的局部，則幾乎沒有任何相似度可言。
> > > 
> > >     - 先利用 Feature Detector 萃取出物體的邊界。
> > > 
> > >     - 再使用 Relu 函數去掉負值，更能淬煉出物體的形狀。
> > > 
> > > - 線性整流單元 ( Rectified Linear Unit，ReLU ) 的數學原理，能將 feature map 上的所有負數轉為 0。這個技巧可以避免讓 CNN 的運算結果趨近 0 或無限大。
> > > 
> > > </details>
> > 
> > > ### Pooling Layer ( 池化層 )
> > > 
> > > 池化是一個壓縮圖片並保留重要資訊的方法，同時增加模型的平移不變性 ( Translation Invariance )，即使輸入圖像中的特徵稍微移動，池化層仍然能夠識別到相同的特徵。
> > > 
> > > <details>
> > > 
> > > <summary>更多詳細內容</summary>
> > > 
> > > - Max Pooling 主要的好處是具備很好的抗雜訊功能，並且當圖片整個平移幾個 Pixel 的話對判斷上完全不會造成影響，能夠提高模型對物體位置變化的容忍度。
> > > 
> > >     - 池化會在圖片上選取不同窗口 ( window )，並在這個窗口範圍中選擇一個最大值。
> > >         
> > >     - 實務上，邊長為二或三的正方形範圍，搭配兩像素的間隔 ( stride ) 是滿理想的設定。
> > > 
> > > - 原圖經過池化以後，其所包含的像素數量會降為原本的四分之一 (2x2)，但因為池化後的圖片包含了原圖中各個範圍的最大值，它還是保留了每個範圍和各個特徵的相符程度。
> > >     
> > >     - 也就是說，池化後的資訊更專注於圖片中是否存在相符的特徵，而非圖片中哪裡存在這些特徵。
> > >         
> > >     - 這能幫助 CNN 判斷圖片中是否包含某項特徵，而不必分心於特徵的位置。
> > > 
> > > </details>
> > 
> > > ### Fully Connected Layer ( 全連接層 )
> > > 
> > > 全連結層會集合高階層中篩選過的圖片 ( 萃取出的特徵 )，並將這些特徵資訊轉化為投票數。
> > > 
> > > <details>
> > > 
> > > <summary>更多詳細內容</summary>
> > > 
> > > - 攤平 ( Flatten )：擔任卷積層到全連接層之間的橋樑。主要是將多維的輸入，攤平成一維輸出，進行維度的轉換。
> > > 
> > >     - 基本上全連接層的部分就是將之前的結果平坦化之後接到最基本的神經網絡了。
> > > 
> > > - 每當 CNN 判斷一張新的圖片時，這張圖片會先經過許多階層，再抵達全連結層。在投票表決之後，擁有最高票數的選項將成為這張圖片的類別。
> > > 
> > >     - 當我們對全連接層輸入圖片時，它會將所有像素的值當成一個一維清單，清單裡的每個值都可以決定圖片分類結果，不過這場選舉並不全然民主。
> > >     
> > >     - 由於某些值可以更好地判斷目標，這些值可以投的票數會比其他值還多。所有值對不同選項所投下的票數，將會以權重 ( weight ) 或 連結強度 ( connection strength ) 的方式來表示。
> > >
> > > </details>
> >
> > </details>
> 
> > ## Captcha ( 驗證碼 )
> > 
> > CAPTCHA：Completely Automated Public Turing test to tell Computers and Humans Apart，全自動公開圖靈測驗的人機辨識方法，又稱「驗證碼」，是一種區分使用者是機器或人類的公共全自動程式。
> >
> > <details>
> > 
> > <summary>更多詳細內容</summary>
> > 
> > - 人機驗證 ( Captcha ) 是一種挑戰，以回應式的安全驗證機制，證明自己是真人，而不是一台嘗試入侵密碼保護帳戶的電腦。
> > 
> > - 系統會以變形的圖片顯示一系列隨機產生的字母和數字，以及一個文字方塊。只要在文字方塊中輸入圖片所顯示的字元，即可通過測驗，證實您的真人身分。
> > 
> > </details>
> 
> </details>

## 功能

- [x] 讀取驗證碼影像
- [x] 影像切割與標籤分割
- [x] 建置 CNN 模型
- [x] 模型訓練：特徵擷取、池化特徵、類別投票
- [x] 儲存模型結果
- [x] 測試模型效果

## 環境部署

開發環境：Google Collaboratory

> The original file is located at：[Captcha Code (CNN)](https://colab.research.google.com/drive/1oGrs1yX4dUr7zFYTSZebuSQ_DX31K8_7)

### 注意事項

1. 此專案部署時，Colab 環境會開啟一台虛擬機，並且使用 google 雲端硬碟來存取檔案。

    - 因此，執行後 Colab 將要求授權，請求連結到 Google Drive 的權限。

2. 完成 Colab 環境部署後，在 Google Drive 將出現一個 Colab Notebooks 的資料夾。

    - 請將本專案的「image.zip」上傳至 Colab Notebooks 的資料夾之中，以利 Colab 透過 google drive 讀取訓練資料 ( 驗證碼圖片 )。

## Demo

> 完整執行結果顯示於 (.ipynb) 檔：[CNN_(captcha_code).ipynb](CNN_(captcha_code).ipynb)

1. Input image：輸入圖形化驗證碼影像。

    ![Input_image](./assets/images/1.%20Input_image.JPG)

2. Split digits：分割圖片，切出各個數字。

    ![split_digits-1](./assets/images/2.%20split_digits-1.JPG)

    ![split_digits-2](./assets/images/2.%20split_digits-2.JPG)

    ![split_digits-3](./assets/images/2.%20split_digits-3.JPG)

    ![split_digits-4](./assets/images/2.%20split_digits-4.JPG)

    ![split_digits-5](./assets/images/2.%20split_digits-5.JPG)

    ![split_digits-6](./assets/images/2.%20split_digits-6.JPG)

    ![split_digits-list](./assets/images/2.%20split_digits-list.JPG)

3. Training：訓練 CNN 模型。

    - Training and Validation Accuracy Curves

        ![training_accuracy_curve](./assets/images/3.%20training_accuracy_curve.JPG)
    
    - Training and Validation Loss Curves
    
        ![training_loss_curve](./assets/images/3.%20training_loss_curve.JPG)

4. Predictions：使用 CNN 模型，辨識未知影像。

    ![predictions](./assets/images/4.%20predictions.JPG)
