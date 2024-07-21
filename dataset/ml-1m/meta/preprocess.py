# 'movies.dat' ——> [MovieID::Title::Genres]
# 'ratings.dat' ——> [UserID::MovieID::Rating::Timestamp]
# 'users.dat' ——> [UserID::Gender::Age::Occupation::Zip-code]
# target fields = ["User ID", "Gender", "Age", "Occupation", "Zipcode", "Movie ID", "Title", "Film genre", "Label"]
import pandas as pd

THRESHOLD = 4

if __name__ == '__main__':
    # read file
    movies_file = 'movies.dat'
    ratings_file = 'ratings.dat'
    users_file = 'users.dat'

    pd.options.display.max_rows = 15

    uname = ['User ID', 'Gender', 'Age', 'Occupation', 'Zipcode']
    users = pd.read_table(users_file, sep='::', header=None, names=uname, engine='python')
    # print(users.head())

    mname = ['Movie ID', 'Title', 'Genres']
    movies = pd.read_table(movies_file, sep='::', header=None, names=mname, engine='python', encoding='ISO-8859-1')
    movies['Film genre'] = movies['Genres'].apply(lambda x: x.split('|')[0])
    # print(movies.head())

    rnames = ['User ID', 'Movie ID', 'Rating', 'Timestamp']
    ratings = pd.read_table(ratings_file, header=None, sep='::', names=rnames, engine='python')
    ratings['Label'] = (ratings['Rating'] >= THRESHOLD).astype(int)
    # print(ratings.head())

    data = pd.merge(pd.merge(ratings, users), movies)

    # split data
    train_data = data.sample(frac=0.8, random_state=200)
    valid_test_data = data.drop(train_data.index)

    # filter data： remove users and movies with only zero labels (remove isolated nodes with no edges)
    users_with_only_zero_labels = train_data.groupby('User ID')['Label'].transform('max') == 0
    filtered_user_ids = train_data[users_with_only_zero_labels]['User ID'].unique()
    movies_with_only_zero_labels = train_data.groupby('Movie ID')['Label'].transform('max') == 0
    filtered_movie_ids = train_data[movies_with_only_zero_labels]['Movie ID'].unique()
    train_data = train_data[~train_data['User ID'].isin(filtered_user_ids) & ~train_data['Movie ID'].isin(filtered_movie_ids)]

    unique_user_ids = train_data['User ID'].unique()
    unique_movie_ids = train_data['Movie ID'].unique()
    filtered_valid_test_data = valid_test_data[
        valid_test_data['User ID'].isin(unique_user_ids) &
        valid_test_data['Movie ID'].isin(unique_movie_ids)
        ]
    valid_test_data = filtered_valid_test_data

    valid_data = valid_test_data.sample(frac=0.5, random_state=200)
    test_data = valid_test_data.drop(valid_data.index)

    print('ml-1m dataset\'s shape:')
    print('\ttrain:', train_data.shape)
    print('\tvalid:', valid_data.shape)
    print('\ttest:', test_data.shape)

    # output file
    train_data.to_csv('../train.csv', index=False)
    valid_data.to_csv('../valid.csv', index=False)
    test_data.to_csv('../test.csv', index=False)
