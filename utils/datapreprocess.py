import pandas as pd
from utils.time_func import timer



class DataPreprocess:
    def __init__(self) -> None:
        self.total_dataframe=pd.DataFrame()
        pass
    
    @timer
    def combime_data_set(self):
        members = pd.read_csv('./data/members.csv')
        songs = pd.read_csv('./data/songs.csv')
        train = pd.read_csv('./data/train.csv')
        test  =pd.read_csv('./data/test.csv')
        train = pd.merge(train, songs, on='song_id', how='left')
        train = pd.merge(train,members,on='msno',how='left')
        test = pd.merge(test, songs, on='song_id', how='left')
        test = pd.merge(test,members,on='msno',how='left')
        return train,test
        
    @timer
    def __fill_dataframe_na(self,df)-> pd.DataFrame:
        
        for col in df.select_dtypes(include=['object']).columns:
            df[col][df[col].isnull()] = 'unknown'
        df = df.fillna(value=0)
        
        return df
    @timer   
    def __registration_init_time_transform(self,df)->pd.DataFrame:
        """
        將時間屬性拆成年、月、日三個屬性
        """
        df.registration_init_time = pd.to_datetime(df.registration_init_time, format='%Y%m%d', errors='ignore')
        df['registration_init_time_year'] = df['registration_init_time'].dt.year
        df['registration_init_time_month'] = df['registration_init_time'].dt.month
        df['registration_init_time_day'] = df['registration_init_time'].dt.day
        df =df.drop(['registration_init_time'], 1)
        return df
    @timer
    def __expiration_date_transform(self,df)->pd.DataFrame:
        """
        將時間屬性拆成年、月、日三個屬性
        """
        df.expiration_date = pd.to_datetime(df.expiration_date,  format='%Y%m%d', errors='ignore')
        df['expiration_date_year'] = df['expiration_date'].dt.year
        df['expiration_date_month'] = df['expiration_date'].dt.month
        df['expiration_date_day'] = df['expiration_date'].dt.day
        df = df.drop(['expiration_date'], 1)
        return df
    
    @timer
    def __obj_to_category(self,df)->pd.DataFrame:
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category')
        return df
    @timer
    def __catagory_encoding(self,df)->pd.DataFrame:
        for col in df.select_dtypes(include=['category']).columns:
            df[col] = df[col].cat.codes
        return df


    def data_pipeline_execute(self,df)->pd.DataFrame:
        df = self.__fill_dataframe_na(df)
        df = self.__registration_init_time_transform(df)
        df = self.__expiration_date_transform(df)
        df = self.__obj_to_category(df)
        df = self.__catagory_encoding(df)
        return df




    


    


    


    






    
        
