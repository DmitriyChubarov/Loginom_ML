import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def one(df):
    df['Услуга международных звонков'] = df['Услуга международных звонков'].apply(lambda x: 1 if x == "True" else 0)    # 1 - остался
    df['Услуга VMail'] = df['Услуга VMail'].apply(lambda x: 1 if x == "True" else 0)    # 1 - остался
    return df

def main():
    df = pd.read_csv('/Users/dmitrij/Desktop/Тестовое задание на позицию аналитика/churn.txt', encoding='windows-1251')
    df = df.drop(columns=['Номер телефона', 'Штат'])
    df = one(df)
    df['Уход'] = df['Уход'].apply(lambda x: 0 if str(x) == 'True' else 1)    # 0 - остался  
    return df

def ml(df):
    global model, X_train, X_test, y_train, y_test

    X = df.drop(columns=['Уход'])
    y = df['Уход']
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.6, random_state=42)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train,y_train)
    df_new = pd.read_csv('/Users/dmitrij/Desktop/Тестовое задание на позицию аналитика/new.txt', encoding='windows-1251')
    df_new = one(df_new)
    df_new = df_new.drop(columns=['Номер телефона', 'Штат','Уход'])
    test_new = model.predict(df_new)
    return test_new
    


print(ml(main()))











