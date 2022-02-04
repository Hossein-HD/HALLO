{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "three-animation",
   "metadata": {},
   "source": [
    "|Month|Savings|Spending|\n",
    "|--- |--- |--- |\n",
    "|January|$100|900|\n",
    "|July|750|1000|\n",
    "|December|250|300|\n",
    "|April|400|700|"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "mechanical-speaker",
   "metadata": {},
   "outputs": [],
   "source": [
    "from urllib.request import urlretrieve"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "humanitarian-excuse",
   "metadata": {},
   "outputs": [],
   "source": [
    "urlretrieve('https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv','medical.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "wireless-herald",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "becoming-madison",
   "metadata": {},
   "outputs": [],
   "source": [
    "medical_df = pd.read_csv('medical.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "computational-plenty",
   "metadata": {},
   "outputs": [],
   "source": [
    "medical_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "respective-updating",
   "metadata": {},
   "outputs": [],
   "source": [
    "medical_df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "uniform-default",
   "metadata": {},
   "outputs": [],
   "source": [
    "medical_df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "wound-planner",
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install plotly"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "loose-banner",
   "metadata": {},
   "outputs": [],
   "source": [
    "import plotly.express as px\n",
    "import matplotlib\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "twenty-check",
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.set_style('darkgrid')\n",
    "matplotlib.rcParams['font.size'] = 14\n",
    "matplotlib.rcParams['figure.figsize'] = (15, 10)\n",
    "matplotlib.rcParams['figure.facecolor'] = '#00000000'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "injured-father",
   "metadata": {},
   "outputs": [],
   "source": [
    "medical_df.age.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "historic-driver",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig = px.histogram (medical_df,\n",
    "                   x='age',\n",
    "                   marginal='box',\n",
    "                    nbins=10,\n",
    "                    title='Distribution of Age')\n",
    "fig.update_layout(bargap = 0.2)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "equipped-northern",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig = px.histogram (medical_df,\n",
    "                   x='bmi',\n",
    "                   marginal='box',\n",
    "                    color='sex',\n",
    "                    color_discrete_sequence = ['red','blue'],\n",
    "                    title='Distribution of BMI')\n",
    "fig.update_layout(bargap = 0.2)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "incorporate-murder",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig = px.histogram (medical_df,\n",
    "                   x='charges',\n",
    "                   marginal='box',\n",
    "                    color = 'smoker',\n",
    "                    color_discrete_sequence = ['red','gray'],\n",
    "                    title='Distribution of BMI')\n",
    "fig.update_layout(bargap = 0.2)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "animated-healthcare",
   "metadata": {},
   "outputs": [],
   "source": [
    "medical_df.smoker.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "developmental-tract",
   "metadata": {},
   "outputs": [],
   "source": [
    "medical_df.sex.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "faced-marijuana",
   "metadata": {},
   "outputs": [],
   "source": [
    "px.histogram(medical_df,x='smoker',color='sex')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "solved-louisville",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig = px.scatter(medical_df,\n",
    "                   x='age',\n",
    "                   y='charges',\n",
    "                    color = 'smoker',\n",
    "                 opacity = 0.8,\n",
    "                 hover_data=['sex'],\n",
    "                    title='Age vs. Charges')\n",
    "fig.update_layout(bargap = 0.2)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "encouraging-hawaiian",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig = px.scatter (medical_df,\n",
    "                 x= 'bmi',\n",
    "                 y= 'charges',\n",
    "                 color= 'smoker',\n",
    "                 opacity= 0.8,\n",
    "                 hover_data=['sex'],\n",
    "                 title = 'Smoking kills')\n",
    "fig.update_traces(marker_size = 5)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "numerical-irrigation",
   "metadata": {},
   "outputs": [],
   "source": [
    "medical_df.charges.corr(medical_df.children)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "accessible-cancer",
   "metadata": {},
   "outputs": [],
   "source": [
    "smoker_values = {'no':0 , 'yes':1}\n",
    "smoker_numeric = medical_df.smoker.map(smoker_values)\n",
    "medical_df [['smoker_code']] = medical_df.smoker.map(smoker_values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "floral-police",
   "metadata": {},
   "outputs": [],
   "source": [
    "sex_values = {'female':0,'male':1}\n",
    "medical_df[['sex_code']] = medical_df.sex.map(sex_values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "silent-captain",
   "metadata": {},
   "outputs": [],
   "source": [
    " medical_df.corr()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "according-advertiser",
   "metadata": {},
   "outputs": [],
   "source": [
    "medical_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ancient-newman",
   "metadata": {},
   "outputs": [],
   "source": [
    "sex_values = {'female':0 , 'male':1}\n",
    "sex_numeric = medical_df.sex.map(sex_values)\n",
    "medical_df.charges.corr(sex_numeric)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "collectible-salem",
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.heatmap(medical_df.corr(),cmap='Blues',annot=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "downtown-honor",
   "metadata": {},
   "outputs": [],
   "source": [
    "non_smoker_df = medical_df[medical_df.smoker == 'no']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "mineral-patrick",
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.scatterplot(data = non_smoker_df , x = 'age', y='charges', alpha = 0.7 , s= 15);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "optimum-developer",
   "metadata": {},
   "outputs": [],
   "source": [
    "def estimate_charges (age,w,b):\n",
    "    return w*age+b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "continued-printing",
   "metadata": {},
   "outputs": [],
   "source": [
    "w=70\n",
    "b=100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "mathematical-confusion",
   "metadata": {},
   "outputs": [],
   "source": [
    "ages = non_smoker_df.age"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "public-sample",
   "metadata": {},
   "outputs": [],
   "source": [
    "ages"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "maritime-albuquerque",
   "metadata": {},
   "outputs": [],
   "source": [
    "estimated_charges = estimate_charges(ages,w,b)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "female-identification",
   "metadata": {},
   "outputs": [],
   "source": [
    "estimated_charges"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "looking-cartoon",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.plot(ages,estimated_charges,'r');\n",
    "plt.scatter(data = non_smoker_df, x='age',y='charges',s= 10,alpha=0.5)\n",
    "plt.legend(['Estimated','Actual'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cognitive-eating",
   "metadata": {},
   "outputs": [],
   "source": [
    "def try_wb(w ,b):\n",
    "    ages=non_smoker_df.age\n",
    "    target=non_smoker_df.charges\n",
    "    estimated_charges = estimate_charges(ages, w, b)\n",
    "    plt.plot(ages,estimated_charges,'r');\n",
    "    plt.scatter(ages,target,s=10,alpha=0.8);\n",
    "    plt.xlabel('Age');\n",
    "    plt.ylabel('Charges');\n",
    "    plt.legend(['Estimated','Actual'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "arctic-smith",
   "metadata": {},
   "outputs": [],
   "source": [
    "try_wb(300,-4000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "complimentary-claim",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "junior-updating",
   "metadata": {},
   "outputs": [],
   "source": [
    "def rmse (targets, predictions):\n",
    "    return np.sqrt(np.mean(np.square(targets-predictions)))\n",
    "\n",
    "def try_wb(w ,b):\n",
    "    ages=non_smoker_df.age\n",
    "    target=non_smoker_df.charges\n",
    "    estimated_charges = estimate_charges(ages, w, b)\n",
    "    plt.plot(ages,estimated_charges,'r');\n",
    "    plt.scatter(ages,target,s=10,alpha=0.8);\n",
    "    plt.xlabel('Age');\n",
    "    plt.ylabel('Charges');\n",
    "    plt.legend(['Estimated','Actual']);\n",
    "    loss = rmse(target,estimated_charges)\n",
    "    print('RMSE: ',loss)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "spectacular-recipient",
   "metadata": {},
   "outputs": [],
   "source": [
    "try_wb(221,-270)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "documentary-kruger",
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install scikit_learn --quiet"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "accessory-potential",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LinearRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "lined-photograph",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = LinearRegression()\n",
    "help(model.fit)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "skilled-dominican",
   "metadata": {},
   "outputs": [],
   "source": [
    "inputs = non_smoker_df[['age']]\n",
    "targets = non_smoker_df.charges\n",
    "targets.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "worse-musical",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.fit(inputs,targets)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "included-logistics",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.predict(np.array([[23],[48]]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "psychological-compiler",
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions = model.predict(inputs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "waiting-missile",
   "metadata": {},
   "outputs": [],
   "source": [
    "rmse(targets,predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "alert-hygiene",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.coef_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "absent-stuart",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.intercept_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "latin-confirmation",
   "metadata": {},
   "outputs": [],
   "source": [
    "try_wb(model.coef_,model.intercept_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "sunrise-layer",
   "metadata": {},
   "outputs": [],
   "source": [
    "m = np.mean(non_smoker_df.charges)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "disturbed-funds",
   "metadata": {},
   "outputs": [],
   "source": [
    "d = np.std(non_smoker_df.charges)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "preceding-draft",
   "metadata": {},
   "outputs": [],
   "source": [
    "U = m+3*d"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "activated-tracker",
   "metadata": {},
   "outputs": [],
   "source": [
    "D = m-3*d"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "separate-taxation",
   "metadata": {},
   "outputs": [],
   "source": [
    "#non_smoker_df.drop(non_smoker_df[non_smoker_df.charges>U].index,inplace=True)\n",
    "#non_smoker_df.drop(non_smoker_df[non_smoker_df.charges<D].index,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "instructional-casino",
   "metadata": {},
   "outputs": [],
   "source": [
    "non_smoker_df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "imported-auction",
   "metadata": {},
   "outputs": [],
   "source": [
    "try_wb(model.coef_,model.intercept_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "endangered-concern",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import SGDRegressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "democratic-reservation",
   "metadata": {},
   "outputs": [],
   "source": [
    "model0= SGDRegressor()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "environmental-justice",
   "metadata": {},
   "outputs": [],
   "source": [
    "help(model0.fit)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "southwest-melbourne",
   "metadata": {},
   "outputs": [],
   "source": [
    "model0.fit(inputs,targets)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "norwegian-horizontal",
   "metadata": {},
   "outputs": [],
   "source": [
    "try_wb(model0.coef_,model0.intercept_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "discrete-gibraltar",
   "metadata": {},
   "outputs": [],
   "source": [
    "inputs, targets = non_smoker_df[['age','bmi','children']], non_smoker_df['charges']\n",
    "model.fit(inputs,targets)\n",
    "rmse(targets,model.predict(inputs))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "another-solid",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.coef_,model.intercept_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "linear-death",
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.heatmap(non_smoker_df.corr(),cmap='Blues',annot=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cooperative-chile",
   "metadata": {},
   "outputs": [],
   "source": [
    "non_smoker_df.age.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "apparent-quantity",
   "metadata": {},
   "outputs": [],
   "source": [
    "targets.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "historic-overhead",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.scatter(non_smoker_df.age,targets,s=10,alpha=0.8)\n",
    "plt.scatter(non_smoker_df.age,model.predict(inputs),cmap='Reds',s=10,alpha=0.8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "nervous-opening",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn import preprocessing\n",
    "enc = preprocessing.OneHotEncoder()\n",
    "enc.fit(medical_df[['region']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "julian-pillow",
   "metadata": {},
   "outputs": [],
   "source": [
    "enc.categories_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "alone-dylan",
   "metadata": {},
   "outputs": [],
   "source": [
    "medical_df[['northeast','northwest','southeast','southwest']] = enc.transform(medical_df[['region']]).toarray()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "changing-frederick",
   "metadata": {},
   "outputs": [],
   "source": [
    "medical_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "spectacular-encyclopedia",
   "metadata": {},
   "outputs": [],
   "source": [
    "inputs,targets = medical_df[['age','bmi','children','somoker_code','sex_code','northeast','northwest','southeast','southwest']],medical_df.charges"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "joint-miniature",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.fit(inputs,targets)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "painted-bonus",
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions = model.predict(inputs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "tamil-throat",
   "metadata": {},
   "outputs": [],
   "source": [
    "rmse(targets , predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "executed-produce",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = preprocessing.StandardScaler()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "separate-italian",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler.fit(medical_df[['age','bmi','children']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "advisory-resolution",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler.mean_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "unlimited-johns",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler.var_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "artificial-iceland",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaled = scaler.transform(medical_df[['age','bmi','children']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "intellectual-xerox",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaled"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "tender-language",
   "metadata": {},
   "outputs": [],
   "source": [
    "other = medical_df[['somoker_code','sex_code','northeast','northwest','southeast','southwest']].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "subsequent-comedy",
   "metadata": {},
   "outputs": [],
   "source": [
    "other"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "subjective-ghana",
   "metadata": {},
   "outputs": [],
   "source": [
    "new_df = np.concatenate((scaled,other),axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "primary-details",
   "metadata": {},
   "outputs": [],
   "source": [
    "new_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "uniform-dallas",
   "metadata": {},
   "outputs": [],
   "source": [
    "targets = medical_df.charges"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "neural-nomination",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = LinearRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ahead-forge",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.fit(new_df,targets)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "flush-exhibition",
   "metadata": {},
   "outputs": [],
   "source": [
    "predict = model.predict(new_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "academic-blackjack",
   "metadata": {},
   "outputs": [],
   "source": [
    "rmse(targets , predict)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "standing-operations",
   "metadata": {},
   "outputs": [],
   "source": [
    " model.coef_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "annoying-guard",
   "metadata": {},
   "outputs": [],
   "source": [
    "new_customer = [[28,30,2,1,0,0,1,0,0]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "italian-virginia",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler.transform([[28,30,2]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "reasonable-malaysia",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.predict([[-0.79795355, -0.10882659,  0.75107928,1,0,0,1,0,0]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "british-horror",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
