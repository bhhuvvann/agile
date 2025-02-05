{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Spliting Dataset\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   GENDER  AGE  SMOKING  YELLOW_FINGERS  ANXIETY  PEER_PRESSURE  \\\n",
      "0       1   69        1               2        2              1   \n",
      "1       1   74        2               1        1              1   \n",
      "2       0   59        1               1        1              2   \n",
      "3       1   63        2               2        2              1   \n",
      "4       0   63        1               2        1              1   \n",
      "\n",
      "   CHRONIC DISEASE  FATIGUE   ALLERGY   WHEEZING  ALCOHOL CONSUMING  COUGHING  \\\n",
      "0                1         2         1         2                  2         2   \n",
      "1                2         2         2         1                  1         1   \n",
      "2                1         2         1         2                  1         2   \n",
      "3                1         1         1         1                  2         1   \n",
      "4                1         1         1         2                  1         2   \n",
      "\n",
      "   SHORTNESS OF BREATH  SWALLOWING DIFFICULTY  CHEST PAIN  LUNG_CANCER  \n",
      "0                    2                      2           2            1  \n",
      "1                    2                      2           2            1  \n",
      "2                    2                      1           2            0  \n",
      "3                    1                      2           2            0  \n",
      "4                    2                      1           1            0  \n"
     ]
    }
   ],
   "source": [
    "data = pd.read_csv(\"/Users/bhuvanm/Downloads/lungcancer.csv\")\n",
    "print(data.head())\n",
    "X = data.drop(columns=['LUNG_CANCER'])  # Replace 'target' with actual target column name\n",
    "y = data['LUNG_CANCER']  # Replace 'target' with actual target column name\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Naive Bayes Accuracy: 94.000000%\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Standardize the dataset\n",
    "scaler = StandardScaler()\n",
    "X_train = scaler.fit_transform(X_train)\n",
    "X_test = scaler.transform(X_test)\n",
    "\n",
    "# Naive Bayes\n",
    "nb = GaussianNB()\n",
    "nb.fit(X_train, y_train)\n",
    "y_pred_nb = nb.predict(X_test)\n",
    "accuracy_nb = accuracy_score(y_test, y_pred_nb)\n",
    "print(f'Naive Bayes Accuracy: {accuracy_nb * 100:f}%')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Naive Bayes "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>GENDER</th>\n",
       "      <th>AGE</th>\n",
       "      <th>SMOKING</th>\n",
       "      <th>YELLOW_FINGERS</th>\n",
       "      <th>ANXIETY</th>\n",
       "      <th>PEER_PRESSURE</th>\n",
       "      <th>CHRONIC DISEASE</th>\n",
       "      <th>FATIGUE</th>\n",
       "      <th>ALLERGY</th>\n",
       "      <th>WHEEZING</th>\n",
       "      <th>ALCOHOL CONSUMING</th>\n",
       "      <th>COUGHING</th>\n",
       "      <th>SHORTNESS OF BREATH</th>\n",
       "      <th>SWALLOWING DIFFICULTY</th>\n",
       "      <th>CHEST PAIN</th>\n",
       "      <th>LUNG_CANCER</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>69</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>74</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>59</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>63</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>63</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>304</th>\n",
       "      <td>0</td>\n",
       "      <td>56</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>305</th>\n",
       "      <td>1</td>\n",
       "      <td>70</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>306</th>\n",
       "      <td>1</td>\n",
       "      <td>58</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>307</th>\n",
       "      <td>1</td>\n",
       "      <td>67</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>308</th>\n",
       "      <td>1</td>\n",
       "      <td>62</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>309 rows × 16 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     GENDER  AGE  SMOKING  YELLOW_FINGERS  ANXIETY  PEER_PRESSURE  \\\n",
       "0         1   69        1               2        2              1   \n",
       "1         1   74        2               1        1              1   \n",
       "2         0   59        1               1        1              2   \n",
       "3         1   63        2               2        2              1   \n",
       "4         0   63        1               2        1              1   \n",
       "..      ...  ...      ...             ...      ...            ...   \n",
       "304       0   56        1               1        1              2   \n",
       "305       1   70        2               1        1              1   \n",
       "306       1   58        2               1        1              1   \n",
       "307       1   67        2               1        2              1   \n",
       "308       1   62        1               1        1              2   \n",
       "\n",
       "     CHRONIC DISEASE  FATIGUE   ALLERGY   WHEEZING  ALCOHOL CONSUMING  \\\n",
       "0                  1         2         1         2                  2   \n",
       "1                  2         2         2         1                  1   \n",
       "2                  1         2         1         2                  1   \n",
       "3                  1         1         1         1                  2   \n",
       "4                  1         1         1         2                  1   \n",
       "..               ...       ...       ...       ...                ...   \n",
       "304                2         2         1         1                  2   \n",
       "305                1         2         2         2                  2   \n",
       "306                1         1         2         2                  2   \n",
       "307                1         2         2         1                  2   \n",
       "308                1         2         2         2                  2   \n",
       "\n",
       "     COUGHING  SHORTNESS OF BREATH  SWALLOWING DIFFICULTY  CHEST PAIN  \\\n",
       "0           2                    2                      2           2   \n",
       "1           1                    2                      2           2   \n",
       "2           2                    2                      1           2   \n",
       "3           1                    1                      2           2   \n",
       "4           2                    2                      1           1   \n",
       "..        ...                  ...                    ...         ...   \n",
       "304         2                    2                      2           1   \n",
       "305         2                    2                      1           2   \n",
       "306         2                    1                      1           2   \n",
       "307         2                    2                      1           2   \n",
       "308         1                    1                      2           1   \n",
       "\n",
       "     LUNG_CANCER  \n",
       "0              1  \n",
       "1              1  \n",
       "2              0  \n",
       "3              0  \n",
       "4              0  \n",
       "..           ...  \n",
       "304            1  \n",
       "305            1  \n",
       "306            1  \n",
       "307            1  \n",
       "308            1  \n",
       "\n",
       "[309 rows x 16 columns]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Decission Tree "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Decision Tree Accuracy: 96.000000%\n"
     ]
    }
   ],
   "source": [
    "dt = DecisionTreeClassifier()\n",
    "dt.fit(X_train, y_train)\n",
    "y_pred_dt = dt.predict(X_test)\n",
    "accuracy_dt = accuracy_score(y_test, y_pred_dt)\n",
    "print(f'Decision Tree Accuracy: {accuracy_dt * 100:f}%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Accuracy: 96.000000%\n"
     ]
    }
   ],
   "source": [
    "rf = RandomForestClassifier(n_estimators=100)\n",
    "rf.fit(X_train, y_train)\n",
    "y_pred_rf = rf.predict(X_test)\n",
    "accuracy_rf = accuracy_score(y_test, y_pred_rf)\n",
    "print(f'Random Forest Accuracy: {accuracy_rf * 100:f}%')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "KNN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "KNN Accuracy: 92.000000%\n"
     ]
    }
   ],
   "source": [
    "knn = KNeighborsClassifier(n_neighbors=5)\n",
    "knn.fit(X_train, y_train)\n",
    "y_pred_knn = knn.predict(X_test)\n",
    "accuracy_knn = accuracy_score(y_test, y_pred_knn)\n",
    "print(f'KNN Accuracy: {accuracy_knn * 100:f}%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "SVM\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SVM Accuracy: 96.000000%\n"
     ]
    }
   ],
   "source": [
    "svm = SVC(kernel='linear')\n",
    "svm.fit(X_train, y_train)\n",
    "y_pred_svm = svm.predict(X_test)\n",
    "accuracy_svm = accuracy_score(y_test, y_pred_svm)\n",
    "print(f'SVM Accuracy: {accuracy_svm * 100:f}%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Logistic Regression \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic Regression Accuracy: 96.000000%\n"
     ]
    }
   ],
   "source": [
    "lr = LogisticRegression()\n",
    "lr.fit(X_train, y_train)\n",
    "y_pred_lr = lr.predict(X_test)\n",
    "accuracy_lr = accuracy_score(y_test, y_pred_lr)\n",
    "print(f'Logistic Regression Accuracy: {accuracy_lr * 100:f}%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Ploting Accuracy "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/anaconda3/lib/python3.11/site-packages/seaborn/_oldcore.py:1765: FutureWarning: unique with argument that is not not a Series, Index, ExtensionArray, or np.ndarray is deprecated and will raise in a future version.\n",
      "  order = pd.unique(vector)\n"
     ]
    },
    {
     "data": {
      "image/png": "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/lq5cqWmTp2qjBkzKlWqVLp3757y5cunjh07asSIEc+9n6d/OfDqevDggW7cuKE8efIoKChI6dKlU0REhN5//339/vvvCgkJUalSpeTj46OPPvpIPj4+evfddzVp0iR7l46X7Hlzy6Ojo7Vq1SqNHTtWGTJk0MKFC+Xp6Wm9febMmapZs6Zy5syZwNUCABKLPn36aPny5SpRooRSpkyphQsXKiAgQO+8845WrFihPn36KFWqVDLGKEOGDNq3b5/OnDmj+vXr6/vvv1e+fPns/RQSDMMPidjvv/+uu3fvqmzZspL+b7re/fv3ValSJb3zzjtq0qSJ6tSpoyFDhmj9+vX6/ffflT9//mfui9D06ouKilLKlCmVJ08e7du3T127dpWfn5/ee+89jRo1Stu3b1dUVJSaN28uT09PWSwWFShQgIPiZCAmNP3888/asGGD7t27p9KlS6t169Zq3LixjDH64osv5Ovrq2+//Vbp0qWTJHXq1MnOlQMA7GnNmjVavHix1qxZo1KlSmnjxo1auHChIiIilC5dOnXo0EE1a9bUiRMn5OzsrCpVqsjR0VHffvutUqVKZZ3JkFwQnBKx/Pnzy9/fX5K0ZMkSeXp6qn379mrfvr3mzJmjXbt2qX79+urUqZMcHBx08eJFnTx58rnBCa++p5eQzp8/v4wxmjBhglxdXVWzZk29+eab1tuDg4M1ceJE7dmzR+PHj7dHuUhAFotFq1atUsuWLVW1alUZY9S1a1dt2bJFI0aMUJMmTRQVFaWZM2eqbt26WrdunTU8AQCSn5gP3K5du6Zq1aqpVKlSWrFihT788EPNnDlTLVq0UHBwsO7evatcuXIpR44ckqRTp05p4sSJWrVqlbZt26aMGTPa+ZkkLIYhEjljjG7cuKGxY8dq4sSJWrdunaQnFxybO3eutmzZomvXrum3337T6dOntXDhQjtXjJchZkbtwYMHtX//fqVJk0bbtm2Tq6urhg8frvXr11tXwlm/fr26d++uefPmadOmTclqCD25iDmZN2a/uHr1qvz8/DR+/HitXbtW69at0549e/Trr7/qs88+kzFGTZo0UevWrZU6dWqFhobas3wAgB1ERERYFwuKmdp9//593b17V8uXL1fbtm01btw4dejQQZK0bt06jRkzRvfv37duf+3aNbm5uemXX35RkSJF7PNE7IjglMhZ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GjlypPbv36+GDRuqX79+1tt69uyp3377TUuWLFG6dOnsWCUSwvnz59W0aVO1a9dOH3/8sYKDg1WgQAFlypRJwcHB6t69uz755BNJ0tKlS+Xj4yNvb287Vw0ASEgbNmyQv7+/Jk6cqClTpujLL7/U3bt35erqqu+//169e/fWm2++qXXr1kmSgoKCFBwcLGdnZ3l7e3O+fDwjOCUirHiSPDwvPI0YMUKTJk3SL7/8EmshALz6/u73OiQkRMOHD1e3bt3k6OioypUrq2rVqurdu7c+/PBDnTp1Sj169JCfn58dqgYAJAZ79uxRkyZNlDp1at28eVPbt2+3Hic8fvxYGzduVO/evVW4cGF99913z2zPsWX8IjgBdhATno4ePaqwsDAdO3ZMu3btUvHixe1dGuJRzB+soKAgXbp0SaGhoapUqZL19kePHsnd3V39+vXThQsXNGfOHKVJk0affvqp1q1bp8yZM2vNmjXy9PRkJBoAkhFjjIwxcnBwUMeOHeXv76+qVavqiy++UIECBaz9wsLCtGHDBvXr10+ZM2fWL7/8Yseqkz4iKGAHMYuD5MmTR3fv3tWePXsITUnM0xc9rlGjht5//301btxYNWvWtPZxd3eX9OTaTa6urkqTJo2kJycBd+nSRevWrVP69OkJTQCQjERHR8tisVhHiqpXr64FCxbo3LlzGjp0qA4cOGDt6+rqqtq1a2v48OHy9PSMtegU4h8jToAd3bp1S9HR0VxfIYl5+qLH5cuXV5cuXdSkSRNt375dffr0Ub9+/TR69GhFRUXJYrFo+PDh2rBhg+rWras7d+5o8eLF2r9/P0uOA0Ay8/TUuqlTp+rPP/9Ujx49lDJlSu3atUutWrVSyZIl1a9fP+sHrt99953q16//3PtA/CI4AcBL8NeLHktPVkbKnz+/ateurW+++cba99ChQ5o5c6Z27typVKlSadasWSpatKidKgcA2MPTC4T16dNHixcv1uDBg1W9enW9/vrrkqQdO3aobdu2Kly4sOrVq6eVK1dq9+7dunXrFmEpAbDEBgDEs6cveuzp6Wlt9/f31927d/X7779r6NChslgs6tixo4oXL67Zs2crNDRUERERSps2rf2KBwAkqMePH8vNzc0amubNm6dvv/1Wa9euValSpSQ9CVUhISGqWLGiFi1apN69e+urr75S6tSpdePGDVZm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eJucOXNq8uTJL60mAEDiQHACALwy2rRpI4vFok6dOj1zW+fOnWWxWNSmTZuELwwAkOQRnAAAr5Ts2bNr6dKlevTokbXt8ePHWrJkiXLkyGHHygAASRnBCQDwSilevLhy5MihVatWWdtWrVql7Nmzq1ixYta2sLAwde/eXRkzZpSbm5sqVKig/fv3x7qvjRs3Kl++fHJ3d1flypV18eLFZx5v9+7deuutt+Tu7q7s2bOre/fuCg0NfWnPDwCQOBGcAACvnA8//FDz5s2zfj937ly1bds2Vp++fftq5cqVWrBggQ4dOqQ8efKoRo0aunv3riTp8uXLeu+991S7dm0dOXJE7du3V//+/WPdx/Hjx1WjRg299957OnbsmAICArRz50517dr15T9JAECiQnACALxyfH19tXPnTl28eFGXLl3Srl271LJlS+vtoaGhmjFjhsaPH69atWqpYMGCmjNnjtzd3eXv7y9JmjFjhl5//XV98cUXeuONN/TBBx88c37U+PHj1aJFC3366afKmzevypUrpy+//FLffPONHj9+nJBPGQBgZ072LgAAgLhKnz696tSpowULFsgYozp16ih9+vTW28+dO6eIiAiVL1/e2ubs7KzSpUvr1KlTkqRTp06pbNmyslgs1j4+Pj6xHufgwYM6e/asFi1aZG0zxig6OloXLlxQgQIFXtZTBAAkMgQnAMArqW3bttYpc1999VWs24wxkhQrFMW0x7TF9Pkn0dHR6tixo7p37/7MbSxEAQDJC1P1AACvpJo1ayo8PFzh4eGqUaNGrNvy5MkjFxcX7dy509oWERGhAwcOWEeJChYsqL1798ba7q/fFy9eXL/99pvy5MnzzJeLi8tLemYAgMSI4AQAeCU5Ojrq1KlTOnXqlBwdHWPdliJFCn388cfq06ePfvjhB508eVIfffSRHj58qHbt2kmSOnXqpHPnzqlnz576448/tHjxYs2fPz/W/fTr10979uxRly5ddOTIEZ05c0Zr165Vt27dEuppAgASCYITAOCVlTp1aqVOnfq5t40ZM0aNGjWSr6+vihcvrrNnz2rTpk167bXXJD2Zardy5UqtW7dORYoU0cyZMzVq1KhY9/G///1P27dv15kzZ1SxYkUVK1ZMgwcPVubMmV/6cwMAJC4W8yKTvAEAAAAgGWPECQAAAABsIDgBAAAAgA0EJwAAAACwgeAEAAAAADYQnAAAAADABoITAAAAANhAcAIAAAAAGwhOAAAAAGADwQkAAAAAbCA4AQAAAIANBCcAAAAAsOH/AUJiUo8vxzmkAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1000x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "accuracy_scores = {\n",
    "    'Naive Bayes': accuracy_nb,\n",
    "    'Decision Tree': accuracy_dt,\n",
    "    'Random Forest': accuracy_rf,\n",
    "    'KNN': accuracy_knn,\n",
    "    'SVM': accuracy_svm,\n",
    "    'Logistic Regression': accuracy_lr\n",
    "}\n",
    "\n",
    "plt.figure(figsize=(10, 5))\n",
    "sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()))\n",
    "plt.xlabel('Model')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.title('Model Performance Comparison')\n",
    "plt.xticks(rotation=45)\n",
    "plt.show()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
