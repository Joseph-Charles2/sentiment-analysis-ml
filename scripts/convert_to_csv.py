import os
import pandas as pd

def load_reviews_from_folder(folder, sentiment):
    reviews = []
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
            reviews.append((f.read(), sentiment))
    return reviews

pos_reviews = load_reviews_from_folder('C:/Users/ADMIN/Downloads/aclImdb_v1/aclImdb/train/pos', 1)
neg_reviews = load_reviews_from_folder('C:/Users/ADMIN/Downloads/aclImdb_v1/aclImdb/train/neg', 0)

# Combine and save to CSV
all_reviews = pos_reviews + neg_reviews
df = pd.DataFrame(all_reviews, columns=['review', 'sentiment'])
df.to_csv('data/reviews.csv', index=False)
