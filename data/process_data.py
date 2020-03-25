# run py process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db in cmd line


# import packages & libraries
import sys
from functools import partial

import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine











def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, on='id', how='left')
    return df

def clean_data(df):
    # create a dataframe of the 36 individual category columns
    #categories = categories.reindex(categories.columns.tolist() + ['col_'+str(i) for i in range(1,37)], axis=1)
    categories = df.categories.str.split(';',expand=True)


    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = [element.split('-')[0] for element in row]

    # rename the columns of `categories`
    categories.columns = category_colnames


    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.split('-')
        categories[column] = categories[column].str[1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    #drop the original categories column
    df = df.drop('categories', axis = 1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()
    #keep only 'related' < 2
    df = df[df['related'] < 2]

    #check: print(type(df.iloc[0,39]))

    return df

def save_data(df, database_filename):
    #save as SQLite database:
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('CleanDataset', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
