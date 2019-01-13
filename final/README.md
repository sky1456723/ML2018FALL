# Testing
使用套件:Pytorch 0.4.1, numpy 1.14.5, pandas 0.23.4.
在執行前，請先確認test.csv、classname.txt在同一folder下,
之後再執行 ```finalscript.sh``` 以跑我們的model.  
此檔案將下載所有不在github上的model, 一一testing並且ensemble.  
shell script用法如： ```bash finalscript.sh "folder of classname.txt and test.csv" "folder of images"```  
例如我可用 "/home/deepq/data/ntu_final_data/medical_images/" 在 "foler of classname.txt and test.csv",  
和 "/home/deepq/data/ntu_final_data/medical_images/images" 在 "folder of images". (on deepq platform).  
如果發生 CUDA out of memory error, 請縮小 "batch_size" 在 "test_combine_model.py" 的第140行  
(只要有一個model發生這個問題，那shell script最後的ensemble.py會出錯，但助教仍可單獨跑"test_combine_model.py"後再跑"ensemble.py")
"test_combine_model.py"用法： ```python test_combine_model.py "model名字" "輸出檔案（請在final底下）" "folder of classname.txt and test.csv" "folder of images"```  
預設情況下會是 model_編號 對應到 model_編號.csv 為其結果，也請助教以此方式命名以免ensemble時找不到file，    
當12個csv都有的時候ensemble.py才不會出錯  
感謝TA們
