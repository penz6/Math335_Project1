Based on the Jupyter Notebook provided, the code runs and is well-structured, but there are a few important data science methodology fixes and code improvements you can make to improve the accuracy and readability of your project.

Here are the recommended fixes:

## **1\. Fix the Exploding MAPE Score (Critical)**

In your calculate\_scores function, you noted that the MAPE value is unreasonably high (3876780865186.898). This is a mathematical artifact of calculating MAPE on **Min-Max scaled data**.

Because Min-Max scaling forces your minimum price to be exactly 0.0, the MAPE formula (which divides by the true value) ends up dividing by zero (or a value extremely close to it), causing the error to explode.

* **The Fix:** You should inverse-transform your y\_pred and y\_test back to their original unscaled values (actual house prices in dollars) before passing them into the calculate\_scores function. This will also make your RMSE and MSE easily interpretable (e.g., an error in dollars rather than a scaled fraction).

Python

\# Assuming you used a separate scaler for the target variable or can isolate it:  
y\_test\_unscaled \= scaler.inverse\_transform(...) \# inverse transform true values  
y\_pred\_unscaled \= scaler.inverse\_transform(...) \# inverse transform predictions  
test\_scores \= calculate\_scores(y\_test\_unscaled, y\_pred\_unscaled)

## **2\. Preprocessing Order: Outlier Removal (Critical)**

In Step 5, you remove outliers *after* applying the MinMaxScaler. This is generally bad practice. Min-Max scaling is highly sensitive to outliers because the minimum and maximum values define the \[0, 1\] boundaries. If you scale first, the outliers squash the rest of your normal data into a tiny range. When you later drop those outliers, your remaining data no longer spans the full 0 to 1 range and remains unnecessarily compressed.

* **The Fix:** Calculate the IQR and remove outliers from the **original, unscaled dataframe** first. *Then*, apply the MinMaxScaler to the cleaned data so that your normal data points utilize the full \[0, 1\] range.

## **3\. Inconsistent Dropout in DNN\_Regressor (Logical)**

In your DNN\_Regressor function, the way you apply Dropout is inconsistent depending on the number of layers:

Python

       for i in range(len(layers)):  
            if len(layers) \== 1:  
                model.add(Dense(layers\[i\], input\_dim=input\_features, activation='relu'))  
                model.add(Dropout(0.10))  
            else:  
                if i \== 0:  
                    model.add(Dense(layers\[i\], input\_dim=input\_features, activation='relu'))  
                    \# MISSING DROPOUT HERE  
                else:  
                    model.add(Dense(layers\[i\], activation='relu'))  
                    model.add(Dropout(0.10))

If you pass multiple layers (e.g., \[20, 10\]), the first hidden layer receives no dropout, while the second one does. If you pass a single layer (e.g., \[10\]), it does receive dropout.

* **The Fix:** Add model.add(Dropout(0.10)) immediately following the if i \== 0: block to ensure uniform regularization, unless intentionally omitting it on the input layer.

## **4\. Code Style & Magic Numbers (Minor)**

In your DNN\_Hyperparameter\_Tuning function, you initialize your best score tracker like this:

Python

best\_avg\_rmse \= 99999999999 \# start with a big number

* **The Fix:** In Python, it is safer and cleaner to use infinity for this:

Python

best\_avg\_rmse \= float('inf')

## **5\. Typos (Minor)**

There are a few minor spelling errors in function names and print statements that you might want to clean up before a final submission:

* def DNN\_Regressor\_Implimentation(...) \-\> Change to DNN\_Regressor\_Implementation.  
* print("Performaing Normality Tests \\n ") \-\> Change to "Performing".  
* \#use colums providid by df itself... \-\> Change to \#use columns provided by df itself...

