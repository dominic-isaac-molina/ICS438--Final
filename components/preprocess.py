import pandas as pd 
import os 
import re 

# Path connection
data_folder = os.path.join(os.path.dirname(__file__), '..', 'raw-data')
ouput_folder = os.path.join(os.path.dirname(__file__), '..', 'output')

train_file = os.path.join(data_folder, 'train.csv')
test_file = os.path.join(data_folder, 'test.csv')
output_file = os.path.join(ouput_folder, 'metadata.parquet')

def clean_text(text):

    # handles empty strings 
    if not isinstance(text, str):
        return ""
    
    # converts everything to lowercase 
    text = text.lower()
    
    # remove special characters but keep numbers and text
    text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
    
    # removes extra spaces so that its just a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_and_preprocess():

    # make sure that the imports are working and connected
    if not os.path.exists(ouput_folder):
        os.makedirs(ouput_folder)

    
    try:
        # Define column names based on the dataset
        col_names = ['class_index', 'title', 'description']
        train_df = pd.read_csv(train_file, names=col_names, header=0)
        test_df = pd.read_csv(test_file, names=col_names, header=0)
        
        # Combine datasets for easier processing
        full_df = pd.concat([train_df, test_df], ignore_index=True)


    except FileNotFoundError:
        # if paths are not working throw an error
        print(f"Error unable to connect to the {data_folder}")
        return

    # cluster using the title
    full_df['raw_text'] = full_df['title']

    # put cleaned data into a df
    full_df['cleaned_text'] = full_df['raw_text'].apply(clean_text)

    
    # Save to metadata.parquet for speed purposes and to keep data types
    print(f"Saving processed data to {output_file}...")
    full_df.to_parquet(output_file, index=False)
    
# connect to main
if __name__ == "__main__": load_and_preprocess()