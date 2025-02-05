{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "872c1449",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions: [1 1 0 1 0 1 0 0 1 0 1 0 1 1 0 1 0 1 1 0 1 0 1 0 1 1 1 1 0 1 0 1 1 1 1 1 1\n",
      " 1 1 1 0 0 1 1 0 0 0 1 1 0 1 0 1 0 1 1 0 0 1 1 0 0 0 0 0 0 1 0 0 1 1 0 1 1\n",
      " 1 1 0 1 1 1 0 0 0 0 1 0 1 0 0 1 1 0 1 1 1 1 0 0 0 0 0 0 1 1 0 1 0 1 1 1 1\n",
      " 1 1 0 1 1 0 1 1 0 1 1 0 0 0 0 1 1 1 1 1 0 1 0 0 1 0 1 1 0 1 1 1 0 1 1 1 1\n",
      " 0 0 1 0 1 1 0 0 1 1 0 0 1 1 0 0 0 0 0 0 0 1 1 1 1 0 1 1 1 0 1 1 1 0 1 1 1\n",
      " 1 1 1 1 1 1 1 1 1 0 0 1 0 1 1 1 1 1 0 0 1 1 1 1 1 1 0 1 1 0 0 0 0 1 0 1 0\n",
      " 0 1 1 1 1 1 0 1 1 0 1 1 0 0 0 0 1 0 0 1 1 0 0 1 0 0 1 0 1 0 0 0 0 1 0 0 0\n",
      " 1 0 1 1 0 0 0 0 1 0 0 0 1 0 1 0 1 1 1 1 1 0 1 0 0 1 0 1 1 0 1 1 1 1 1 0 0\n",
      " 1 0 1 1 1 0 1 1 0 0 0 1]\n",
      "Accuracy: 81.4935064935065\n"
     ]
    }
   ],
   "source": [
    "#GaussianNB\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.metrics import accuracy_score\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "df = pd.read_csv(\"heart.csv\")\n",
    "x = df.drop('target', axis=1)\n",
    "y = df['target']\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)\n",
    "model = GaussianNB()\n",
    "model.fit(x_train, y_train)\n",
    "y_pred = model.predict(x_test)\n",
    "print(\"Predictions:\", y_pred)\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred) * 100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "e9955cc4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions: [1 1 0 1 0 1 0 0 1 0 1 0 1 1 0 1 0 1 1 0 1 0 1 0 1 1 1 1 0 1 0 1 1 1 1 1 1\n",
      " 1 1 1 0 0 1 1 0 0 0 1 1 0 1 0 1 0 1 1 0 0 1 1 0 0 0 0 0 0 1 0 0 1 1 0 1 1\n",
      " 1 1 0 1 1 1 0 0 0 0 1 0 1 0 0 1 1 0 1 1 1 1 0 0 0 0 0 0 1 1 0 1 0 1 1 1 1\n",
      " 1 1 0 1 1 0 1 1 0 1 1 0 0 0 0 1 1 1 1 1 0 1 0 0 1 0 1 1 0 1 1 1 0 1 1 1 1\n",
      " 0 0 1 0 1 1 0 0 1 1 0 0 1 1 0 0 0 0 0 0 0 1 1 1 1 0 1 1 1 0 1 1 1 0 1 1 1\n",
      " 1 1 1 1 1 1 1 1 1 0 0 1 0 1 1 1 1 1 0 0 1 1 1 1 1 1 0 1 1 0 0 0 0 1 0 1 0\n",
      " 0 1 1 1 1 1 0 1 1 0 1 1 0 0 0 0 1 0 0 1 1 0 0 1 0 0 1 0 1 0 0 0 0 1 0 0 0\n",
      " 1 0 1 1 0 0 0 0 1 0 0 0 1 0 1 0 1 1 1 1 1 0 1 0 0 1 0 1 1 0 1 1 1 1 1 0 0\n",
      " 1 0 1 1 1 0 1 1 0 0 0 1]\n",
      "Accuracy: 81.4935064935065\n"
     ]
    }
   ],
   "source": [
    "#Decision Tree\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.metrics import accuracy_score\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "df = pd.read_csv(\"heart.csv\")\n",
    "x = df.drop('target', axis=1)\n",
    "y = df['target']\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)\n",
    "model = GaussianNB()\n",
    "model.fit(x_train, y_train)\n",
    "y_pred = model.predict(x_test)\n",
    "print(\"Predictions:\", y_pred)\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred) * 100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "1dd9efd5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions: [1 1 0 1 0 1 0 0 1 0 1 0 1 1 0 0 0 1 1 0 0 0 0 0 0 1 1 1 0 0 0 1 0 1 1 1 0\n",
      " 1 1 1 0 0 1 0 0 0 0 0 0 1 1 0 0 0 1 1 0 0 0 1 1 1 0 1 0 0 1 0 0 1 0 0 0 1\n",
      " 1 1 0 0 0 1 0 0 0 0 1 0 1 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 1 0 1 0 1 0 1 0\n",
      " 1 1 0 1 1 0 1 1 0 1 1 0 0 1 0 1 0 0 1 1 0 1 0 0 1 0 1 1 0 1 1 1 1 1 1 1 1\n",
      " 0 0 0 0 1 1 0 0 0 1 0 0 1 1 0 0 1 1 0 0 1 1 0 1 1 0 1 1 1 0 0 1 1 0 1 0 1\n",
      " 1 1 0 1 1 1 0 0 0 0 1 0 0 1 1 1 1 1 0 0 0 1 1 1 1 1 0 1 0 0 0 0 0 1 0 1 0\n",
      " 0 1 1 1 1 1 0 1 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 1 0 0 1 0 1 0 0 0 0 1 0 1 0\n",
      " 1 0 1 1 0 0 0 0 1 0 0 0 1 0 1 0 1 1 1 0 1 0 1 0 0 1 0 1 1 0 1 1 0 1 1 0 1\n",
      " 1 0 0 1 0 0 1 1 0 0 0 0]\n",
      "Accuracy: 98.05194805194806\n"
     ]
    }
   ],
   "source": [
    "#RandomForest\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "df = pd.read_csv(\"heart.csv\")\n",
    "x = df.drop('target', axis=1)\n",
    "y = df['target']\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)\n",
    "model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "model.fit(x_train, y_train)\n",
    "y_pred = model.predict(x_test)\n",
    "print(\"Predictions:\", y_pred)\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred) * 100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "7585d9e3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions: [1 1 0 1 0 1 0 0 1 0 1 0 1 1 0 1 0 1 1 0 1 0 0 0 1 1 1 1 0 1 1 1 0 1 1 1 1\n",
      " 0 1 1 0 0 1 0 0 0 0 0 1 1 1 0 1 0 1 1 0 0 1 1 0 0 0 1 0 0 1 1 0 1 1 0 0 1\n",
      " 1 1 0 1 1 1 0 0 0 0 1 0 1 0 0 1 1 0 1 1 1 1 1 0 0 0 0 0 1 1 0 1 0 1 0 1 1\n",
      " 1 1 0 1 1 1 1 1 0 0 1 0 0 0 0 1 1 1 1 1 0 1 0 0 1 0 1 1 1 0 1 1 0 1 1 1 1\n",
      " 1 0 0 0 1 0 0 0 1 1 0 0 1 1 0 0 0 0 0 0 0 1 0 1 1 0 1 1 1 0 1 1 1 0 1 1 1\n",
      " 1 1 1 1 1 1 1 1 1 0 0 1 0 1 1 1 1 1 0 0 1 1 1 0 1 1 0 1 1 0 0 1 0 1 0 1 0\n",
      " 0 1 1 1 1 1 0 0 1 0 1 1 0 0 0 0 1 1 0 1 1 0 0 1 0 0 1 0 1 0 0 0 0 1 0 1 0\n",
      " 1 0 0 1 0 0 0 0 1 0 0 0 1 0 1 0 1 1 1 1 1 0 1 0 0 1 0 1 1 0 1 1 0 1 1 0 1\n",
      " 1 0 0 1 1 0 1 1 1 0 0 1]\n",
      "Accuracy: 80.84415584415584\n"
     ]
    }
   ],
   "source": [
    "#SVM\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import accuracy_score\n",
    "df = pd.read_csv(\"heart.csv\")\n",
    "x = df.drop('target', axis=1)\n",
    "y = df['target']\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)\n",
    "classifier = SVC(kernel='linear', random_state=0)\n",
    "classifier.fit(x_train, y_train)\n",
    "y_pred = classifier.predict(x_test)\n",
    "print(\"Predictions:\", y_pred)\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred) * 100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "47916399",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions: [1 1 0 1 0 1 0 0 1 0 1 0 1 1 0 1 0 1 1 0 0 0 1 0 0 1 1 1 1 1 0 0 0 1 1 1 0\n",
      " 0 0 1 0 1 0 1 0 0 0 1 0 1 1 0 1 0 1 0 0 1 1 1 1 1 1 1 0 0 1 0 0 1 0 1 1 1\n",
      " 1 1 0 0 0 1 1 0 0 0 1 0 1 0 1 1 0 0 1 1 1 1 0 0 0 0 1 0 0 0 0 0 1 0 0 0 1\n",
      " 0 1 0 0 1 1 0 1 0 0 1 0 1 1 0 1 0 0 1 0 0 1 1 0 1 0 1 1 1 1 1 1 0 0 0 1 1\n",
      " 1 0 0 0 1 1 0 0 1 1 0 1 0 0 0 0 1 0 1 0 0 1 1 1 1 0 0 1 1 0 0 1 1 0 1 1 1\n",
      " 1 1 1 0 1 1 0 1 1 1 0 0 0 1 1 1 0 1 0 0 1 1 1 1 1 0 0 1 0 0 1 1 1 1 0 0 0\n",
      " 0 1 1 1 0 1 0 0 1 0 0 1 0 0 1 0 1 1 0 0 1 0 0 0 0 1 1 0 1 0 0 0 1 1 0 1 0\n",
      " 0 0 0 1 0 0 0 0 1 1 1 0 1 0 1 1 1 1 1 0 1 0 0 0 1 1 0 1 1 1 1 0 0 1 1 0 0\n",
      " 1 0 1 0 1 0 1 1 1 0 0 0]\n",
      "Accuracy: 71.42857142857143\n"
     ]
    }
   ],
   "source": [
    "#KNN\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.neighbors import KNeighborsClassifier  \n",
    "from sklearn.metrics import accuracy_score\n",
    "df = pd.read_csv(\"heart.csv\")\n",
    "x = df.drop('target', axis=1)\n",
    "y = df['target']\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)\n",
    "classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  \n",
    "classifier.fit(x_train, y_train)\n",
    "y_pred = classifier.predict(x_test)\n",
    "print(\"Predictions:\", y_pred)\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred) * 100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "b4b75721",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions: [1 1 0 1 0 1 0 0 1 0 1 0 1 1 0 1 0 1 1 0 1 0 0 0 1 1 1 1 0 1 1 1 0 1 1 1 1\n",
      " 0 1 1 0 0 1 0 0 0 0 0 1 0 1 0 1 0 1 1 0 0 1 1 1 0 0 0 0 0 1 1 0 1 1 0 0 1\n",
      " 1 1 0 1 1 1 0 0 0 0 1 0 1 0 0 1 1 0 1 1 1 1 1 0 0 0 0 0 1 1 0 1 0 1 0 1 1\n",
      " 1 1 0 1 1 1 1 1 0 0 1 0 0 0 0 1 1 1 1 1 0 1 0 0 1 0 1 1 1 0 1 1 0 1 1 1 1\n",
      " 1 0 0 0 1 0 0 0 1 1 0 0 1 1 0 0 0 0 0 0 0 1 0 1 1 0 1 1 1 0 1 1 1 0 1 1 1\n",
      " 1 1 1 1 1 1 1 1 1 0 0 1 0 1 1 1 1 1 0 0 1 1 1 0 1 1 0 1 1 0 0 1 0 1 0 1 0\n",
      " 0 1 1 1 1 1 0 0 1 0 1 1 0 0 0 0 1 1 0 1 1 0 0 1 0 0 1 0 1 0 0 0 0 1 0 1 0\n",
      " 1 0 0 1 0 0 0 0 1 0 0 0 1 0 1 0 1 1 1 1 1 0 1 0 0 1 0 1 1 0 1 1 0 1 1 0 1\n",
      " 1 0 0 1 1 0 1 1 1 0 0 1]\n",
      "Accuracy: 80.51948051948052\n"
     ]
    }
   ],
   "source": [
    "#LogisticRegression\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score\n",
    "df = pd.read_csv(\"heart.csv\")\n",
    "x = df.drop('target', axis=1)\n",
    "y = df['target']\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)\n",
    "classifier = LogisticRegression(max_iter=1000)\n",
    "classifier.fit(x_train, y_train)\n",
    "y_pred = classifier.predict(x_test)\n",
    "print(\"Predictions:\", y_pred)\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred) * 100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "8e7394f7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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WTAyMkKbNm3w7t071KlTB6tWrYKhoWG6Px8iIiLKeDLMOjv6xHV2iIiIMp9Mv84OERERUWrIsOvsEBGllykXXui7BL0Y6ZZL3yUQpQuO7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqGem7ACIiynymXHih7xL0ZqRbri/eNqvut/+yz1JDhh7ZiY+Px5gxY+Di4gJzc3MUKFAAEyZMQGJiotJHRDBu3Dg4OjrC3NwcNWvWxLVr1/RYNREREWUkGXpkZ+rUqVi8eDFWr16NEiVK4OzZs+jWrRtsbGwwePBgAMC0adMwc+ZMrFq1CkWKFMGkSZNQr1493Lx5E1ZWVnp+BkTpi381EhEll6FHdk6ePIlmzZqhcePGyJ8/P1q1aoX69evj7NmzAD6M6syePRujR49Gy5YtUbJkSaxevRpv375FQECAnqsnIiKijCBDj+xUrVoVixcvxq1bt1CkSBFcunQJx44dw+zZswEA9+7dQ2hoKOrXr69sY2pqiho1auDEiRPo06fPRx83JiYGMTExyu3IyMg0fR6UchyhICKi1JKhw463tzciIiJQrFgxGBoaIiEhAZMnT0b79u0BAKGhoQAAe3t7ne3s7e3x4MGDTz6uv78/xo8fn3aFExERUYaRoQ9jbdq0CevWrUNAQADOnz+P1atXY8aMGVi9erVOP41Go3NbRJK1JeXj44OIiAjl5+HDh2lSPxEREelfhh7ZGT58OEaOHIl27doBAEqVKoUHDx7A398fXbp0gYODA4APIzx58uRRtgsLC0s22pOUqakpTE1N07Z4IiIiyhAy9MjO27dvYWCgW6KhoaFy6rmLiwscHBywf/9+5f7Y2FgEBQXB3d09XWslIiKijClDj+x4enpi8uTJ+Oqrr1CiRAlcuHABM2fORPfu3QF8OHzl5eUFPz8/FC5cGIULF4afnx8sLCzQoUMHPVdPREREGUGGDjvz5s3D2LFj0a9fP4SFhcHR0RF9+vTBjz/+qPQZMWIE3r17h379+uH169eoVKkS9u3bxzV2iIiICEAGDztWVlaYPXu2cqr5x2g0GowbNw7jxo1Lt7qIiIgo88jQc3aIiIiI/iuGHSIiIlI1hh0iIiJSNYYdIiIiUjWGHSIiIlI1hh0iIiJSNYYdIiIiUjWGHSIiIlI1hh0iIiJSNYYdIiIiUjWGHSIiIlI1hh0iIiJSNYYdIiIiUjWGHSIiIlI1hh0iIiJSNYYdIiIiUjWGHSIiIlI1hh0iIiJSNYYdIiIiUjWGHSIiIlI1hh0iIiJSNYYdIiIiUjWGHSIiIlI1hh0iIiJSNSN9F6B2Uy680HcJejPSLZe+SyAiIuLIDhEREakbww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqZpRSjqLCIKCgnD06FHcv38fb9++Re7cueHm5oa6devCyckpreokIiIi+iKfNbLz7t07+Pn5wcnJCQ0bNsTu3bsRHh4OQ0ND3LlzB76+vnBxcUGjRo1w6tSptK6ZiIiI6LN91shOkSJFUKlSJSxevBgeHh4wNjZO1ufBgwcICAhA27ZtMWbMGPTq1SvViyUiIiJKqc8KO3v27EHJkiX/sY+zszN8fHzwww8/4MGDB6lSHBEREdF/9VmHsf4t6CRlYmKCwoULf3FBRERERKkpRROUk4qPj8eSJUtw+PBhJCQkoEqVKujfvz/MzMxSsz4iIiKi/+SLw86gQYNw69YttGzZEnFxcVizZg3Onj2LDRs2pGZ9RERERP/JZ4edwMBAtGjRQrm9b98+3Lx5E4aGhgAADw8PfPPNN6lfIREREdF/8NmLCi5fvhzNmzfH48ePAQDlypXD999/j99//x07d+7EiBEjULFixTQrlIiIiOhLfHbY2bVrF9q1a4eaNWti3rx5WLp0KaytrTF69GiMHTsWTk5OCAgISPUCHz9+jO+++w62trawsLBA2bJlce7cOeV+EcG4cePg6OgIc3Nz1KxZE9euXUv1OoiIiChzStHlItq1a4czZ87g8uXL8PDwQKdOnXDu3DlcvHgRCxYsQO7cuVO1uNevX6NKlSowNjbGnj17EBwcjJ9++gnZs2dX+kybNg0zZ87E/PnzcebMGTg4OKBevXqIiopK1VqIiIgoc0rxBOXs2bNj2bJlOHLkCDp16oQGDRpgwoQJMDc3T/Xipk6dCicnJ6xcuVJpy58/v/L/IoLZs2dj9OjRaNmyJQBg9erVsLe3R0BAAPr06ZPqNREREVHm8tkjOw8fPkTbtm1RqlQpdOzYEYULF8a5c+dgbm6OsmXLYs+ePale3I4dO1ChQgW0bt0adnZ2cHNzw7Jly5T77927h9DQUNSvX19pMzU1RY0aNXDixIlPPm5MTAwiIyN1foiIiEidPjvsdO7cGRqNBtOnT4ednR369OkDExMTTJgwAdu3b4e/vz/atGmTqsX99ddfWLRoEQoXLoy9e/fi+++/x6BBg7BmzRoAQGhoKADA3t5eZzt7e3vlvo/x9/eHjY2N8sMLmBIREanXZx/GOnv2LC5evIiCBQvCw8MDLi4uyn3FixfHkSNHsHTp0lQtLjExERUqVICfnx8AwM3NDdeuXcOiRYvQuXNnpZ9Go9HZTkSStSXl4+ODoUOHKrcjIyMZeIiIiFTqs0d2ypUrhx9//BH79u2Dt7c3SpUqlaxP7969U7W4PHnywNXVVaetePHiCAkJAQA4ODgAQLJRnLCwsGSjPUmZmprC2tpa54eIiIjU6bPDzpo1axATE4MhQ4bg8ePHWLJkSVrWBQCoUqUKbt68qdN269YtODs7AwBcXFzg4OCA/fv3K/fHxsYiKCgI7u7uaV4fERERZXyffRjL2dkZv/zyS1rWksyQIUPg7u4OPz8/tGnTBn/++SeWLl2qHC7TaDTw8vKCn58fChcujMKFC8PPzw8WFhbo0KFDutZKREREGdNnhZ3o6GhYWlp+9oOmtP+nVKxYEYGBgfDx8cGECRPg4uKC2bNno2PHjkqfESNG4N27d+jXrx9ev36NSpUqYd++fbCysvrPv5+IiIgyv886jFWoUCH4+fnhyZMnn+wjIti/fz8aNmyIuXPnplqBTZo0wZUrV/D+/Xtcv34dvXr10rlfo9Fg3LhxePr0Kd6/f4+goCCULFky1X4/ERERZW6fNbJz+PBhjBkzBuPHj0fZsmVRoUIFODo6wszMDK9fv0ZwcDBOnjwJY2Nj+Pj4pPpEZSIiIqIv9Vlhp2jRotiyZQsePXqELVu24MiRIzhx4gTevXuHXLlyKYv9NWrUCAYGKboCBREREVGaStHlIvLly4chQ4ZgyJAhaVUPERERUariMAwRERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqVqKw07+/PkxYcIE5WKcRERERBlZisPODz/8gF9//RUFChRAvXr1sHHjRsTExKRFbURERET/WYrDzsCBA3Hu3DmcO3cOrq6uGDRoEPLkyYMBAwbg/PnzaVEjERER0Rf74jk7ZcqUwZw5c/D48WP4+vri559/RsWKFVGmTBmsWLECIpKadRIRERF9kRStoJxUXFwcAgMDsXLlSuzfvx/ffPMNevTogSdPnmD06NH4448/EBAQkJq1EhEREaVYisPO+fPnsXLlSmzYsAGGhobo1KkTZs2ahWLFiil96tevj+rVq6dqoURERERfIsVhp2LFiqhXrx4WLVqE5s2bw9jYOFkfV1dXtGvXLlUKJCIiIvovUhx2/vrrLzg7O/9jH0tLS6xcufKLiyIiIiJKLSmeoBwWFobTp08naz99+jTOnj2bKkURERERpZYUh53+/fvj4cOHydofP36M/v37p0pRRERERKklxWEnODgY5cqVS9bu5uaG4ODgVCmKiIiIKLWkOOyYmpri2bNnydqfPn0KI6MvPpOdiIiIKE2kOOzUq1cPPj4+iIiIUNrCw8MxatQo1KtXL1WLIyIiIvqvUjwU89NPP6F69epwdnaGm5sbAODixYuwt7fH2rVrU71AIiIiov8ixWEnb968uHz5MtavX49Lly7B3Nwc3bp1Q/v27T+65g4RERGRPn3RJBtLS0v07t07tWshIiIiSnVfPKM4ODgYISEhiI2N1Wlv2rTpfy6KiIiIKLV80QrKLVq0wJUrV6DRaJSrm2s0GgBAQkJC6lZIRERE9B+k+GyswYMHw8XFBc+ePYOFhQWuXbuGI0eOoEKFCjh8+HAalEhERET05VI8snPy5EkcPHgQuXPnhoGBAQwMDFC1alX4+/tj0KBBuHDhQlrUSURERPRFUjyyk5CQgGzZsgEAcuXKhSdPngAAnJ2dcfPmzdStjoiIiOg/SvHITsmSJXH58mUUKFAAlSpVwrRp02BiYoKlS5eiQIECaVEjERER0RdLcdgZM2YMoqOjAQCTJk1CkyZNUK1aNdja2mLTpk2pXiARERHRf5HisOPh4aH8f4ECBRAcHIxXr14hR44cyhlZRERERBlFiubsxMfHw8jICFevXtVpz5kzJ4MOERERZUgpCjtGRkZwdnbmWjpERESUaaT4bKwxY8bAx8cHr169Sot6iIiIiFJViufszJ07F3fu3IGjoyOcnZ1haWmpc//58+dTrTgiIiKi/yrFYad58+ZpUAYRERFR2khx2PH19U2LOoiIiIjSRIrn7BARERFlJike2TEwMPjH08x5phYRERFlJCkOO4GBgTq34+LicOHCBaxevRrjx49PtcKIiIiIUkOKw06zZs2StbVq1QolSpTApk2b0KNHj1QpjIiIiCg1pNqcnUqVKuGPP/5IrYcjIiIiShWpEnbevXuHefPmIV++fKnxcERERESpJsWHsf5+wU8RQVRUFCwsLLBu3bpULY6IiIjov0px2Jk1a5ZO2DEwMEDu3LlRqVIl5MiRI1WLIyIiIvqvUhx2unbtmgZlEBEREaWNFM/ZWblyJbZs2ZKsfcuWLVi9enWqFEVERESUWlIcdqZMmYJcuXIla7ezs4Ofn1+qFEVERESUWlIcdh48eAAXF5dk7c7OzggJCUmVooiIiIhSS4rDjp2dHS5fvpys/dKlS7C1tU2VooiIiIhSS4rDTrt27TBo0CAcOnQICQkJSEhIwMGDBzF48GC0a9cuLWokIiIi+mIpPhtr0qRJePDgAerUqQMjow+bJyYmonPnzpyzQ0RERBlOisOOiYkJNm3ahEmTJuHixYswNzdHqVKl4OzsnBb1EREREf0nKQ47WoULF0bhwoVTsxYiIiKiVJfiOTutWrXClClTkrVPnz4drVu3TpWiiIiIiFJLisNOUFAQGjdunKy9QYMGOHLkSKoURURERJRaUhx23rx5AxMTk2TtxsbGiIyMTJWiiIiIiFJLisNOyZIlsWnTpmTtGzduhKura6oURURERJRaUjxBeezYsfj2229x9+5d1K5dGwBw4MABbNiw4aPXzCIiIiLSpxSHnaZNm2L79u3w8/PDL7/8AnNzc5QuXRp//PEHatSokRY1EhEREX2xFB/GAoDGjRvj+PHjiI6OxosXL3Dw4EHUqFEDFy9eTOXydPn7+0Oj0cDLy0tpExGMGzcOjo6OMDc3R82aNXHt2rU0rYOIiIgyjy8KO0lFRERg4cKFKFeuHMqXL58aNX3UmTNnsHTpUpQuXVqnfdq0aZg5cybmz5+PM2fOwMHBAfXq1UNUVFSa1UJERESZxxeHnYMHD6Jjx47IkycP5s2bh0aNGuHs2bOpWZvizZs36NixI5YtW4YcOXIo7SKC2bNnY/To0WjZsiVKliyJ1atX4+3btwgICEiTWoiIiChzSVHYefToESZNmoQCBQqgffv2yJkzJ+Li4rB161ZMmjQJbm5uaVJk//790bhxY9StW1en/d69ewgNDUX9+vWVNlNTU9SoUQMnTpz45OPFxMQgMjJS54eIiIjU6bPDTqNGjeDq6org4GDMmzcPT548wbx589KyNgAfTmk/f/48/P39k90XGhoKALC3t9dpt7e3V+77GH9/f9jY2Cg/Tk5OqVs0ERERZRifHXb27duHnj17Yvz48WjcuDEMDQ3Tsi4AwMOHDzF48GCsW7cOZmZmn+yn0Wh0botIsrakfHx8EBERofw8fPgw1WomIiKijOWzw87Ro0cRFRWFChUqoFKlSpg/fz6eP3+elrXh3LlzCAsLQ/ny5WFkZAQjIyMEBQVh7ty5MDIyUkZ0/j6KExYWlmy0JylTU1NYW1vr/BAREZE6fXbYqVy5MpYtW4anT5+iT58+2LhxI/LmzYvExETs378/Tc5+qlOnDq5cuYKLFy8qPxUqVEDHjh1x8eJFFChQAA4ODti/f7+yTWxsLIKCguDu7p7q9RAREVHmk+KzsSwsLNC9e3ccO3YMV65cwQ8//IApU6bAzs4OTZs2TdXirKysULJkSZ0fS0tL2NraomTJksqaO35+fggMDMTVq1fRtWtXWFhYoEOHDqlaCxEREWVO/2mdnaJFi2LatGl49OgRNmzYkFo1pciIESPg5eWFfv36oUKFCnj8+DH27dsHKysrvdRDREREGUuKLxfxMYaGhmjevDmaN2+eGg/3jw4fPqxzW6PRYNy4cRg3blya/24iIiLKfP7zCspEREREGRnDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqVqGDjv+/v6oWLEirKysYGdnh+bNm+PmzZs6fUQE48aNg6OjI8zNzVGzZk1cu3ZNTxUTERFRRpOhw05QUBD69++PU6dOYf/+/YiPj0f9+vURHR2t9Jk2bRpmzpyJ+fPn48yZM3BwcEC9evUQFRWlx8qJiIgoozDSdwH/5Pfff9e5vXLlStjZ2eHcuXOoXr06RASzZ8/G6NGj0bJlSwDA6tWrYW9vj4CAAPTp00cfZRMREVEGkqFHdv4uIiICAJAzZ04AwL179xAaGor69esrfUxNTVGjRg2cOHHik48TExODyMhInR8iIiJSp0wTdkQEQ4cORdWqVVGyZEkAQGhoKADA3t5ep6+9vb1y38f4+/vDxsZG+XFyckq7womIiEivMk3YGTBgAC5fvowNGzYku0+j0ejcFpFkbUn5+PggIiJC+Xn48GGq10tEREQZQ4aes6M1cOBA7NixA0eOHEG+fPmUdgcHBwAfRnjy5MmjtIeFhSUb7UnK1NQUpqamaVcwERERZRgZemRHRDBgwABs27YNBw8ehIuLi879Li4ucHBwwP79+5W22NhYBAUFwd3dPb3LJSIiogwoQ4/s9O/fHwEBAfj1119hZWWlzMOxsbGBubk5NBoNvLy84Ofnh8KFC6Nw4cLw8/ODhYUFOnTooOfqiYiIKCPI0GFn0aJFAICaNWvqtK9cuRJdu3YFAIwYMQLv3r1Dv3798Pr1a1SqVAn79u2DlZVVOldLREREGVGGDjsi8q99NBoNxo0bh3HjxqV9QURERJTpZOg5O0RERET/FcMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREamaasLOwoUL4eLiAjMzM5QvXx5Hjx7Vd0lERESUAagi7GzatAleXl4YPXo0Lly4gGrVqqFhw4YICQnRd2lERESkZ6oIOzNnzkSPHj3Qs2dPFC9eHLNnz4aTkxMWLVqk79KIiIhIzzJ92ImNjcW5c+dQv359nfb69evjxIkTeqqKiIiIMgojfRfwX7148QIJCQmwt7fXabe3t0doaOhHt4mJiUFMTIxyOyIiAgAQGRmZ6vW9fxOV6o+ZWURGmnzxtll1v/2XfQZwv30p7reUy6r7DOB++xL/9T366cf98L0tIv/YL9OHHS2NRqNzW0SStWn5+/tj/PjxydqdnJzSpLasKvkepn/DffZluN++DPfbl+F+S7m03mdRUVGwsbH55P2Z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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "accuracies = [71.42857142857143, 80.51948051948052,  80.84415584415584, 98.05194805194806, 81.4935064935065, 81.4935064935065]\n",
    "models = ['KNN', 'Logistic Regression', 'SVM', 'Random Forest', 'Decision Tree', 'Gaussian NB']\n",
    "plt.bar(models, accuracies, color='skyblue')\n",
    "plt.xlabel('Model')\n",
    "plt.ylabel('Accuracy (%)')\n",
    "plt.title('Model Accuracy Comparison')\n",
    "plt.xticks(rotation=45)\n",
    "plt.show()"
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
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
