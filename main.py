from __future__ import print_function
import numpy as np
import pandas as pd
# Reading user file:
from cffi.backend_ctypes import xrange

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols)
n_users = users.shape[0]
print('Number of users:', n_users)
#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols)
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols)
rate_train = ratings_base.values
rate_test = ratings_test.values
print('Number of traing rates:', rate_train.shape[0])
print('Number of test rates:', rate_test.shape[0])

#Công việc quan trọng trong hệ thống gợi ý dựa trên nội dung là xây dựng vector đặc trưng cho mỗi sản phẩm.
# trước hết, chúng ta cần lưu thông tin về các sản phẩm vào biến items:
#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date',
'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s',
'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War','Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols)
n_items = items.shape[0]
print('Number of items:', n_items)

#Ta chỉ quan tâm tới 19 giá trị nhị phân ở cuối mỗi hàng để xây dựng thông tin sản phẩm:
X0 = items.values
X_train_counts = X0[:, -19:]
#Tiếp theo, chúng ta hiển thị một vài hàng đầu tiên của ma trận rate_train:
print(rate_train[:4, :])
#Kết quả: Hàng thứ nhất được hiểu là người dùng thứ nhất đánh giá bộ phim thứ nhất năm sao.
# cột cuối cùng là thời điểm đánh giá, chúng ta sẽ bỏ qua thông số này.


#Tiếp theo, chúng ta sẽ xây dựng vector đặc trưng cho mỗi sản phẩm dựa trên ma trận thẻ loại phim và đặc trưng TF-IDF:

#tfidf
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=True, norm ='l2')
X = transformer.fit_transform(X_train_counts.tolist()).toarray()
#Sau bước này, mỗi hàng của X tương ứng với vector đặc trung của một bộ phim

#Với mỗi người dùng, chúng ta cần đi tìm những bộ phim nào mà người dùng đó đã đánh giá,
# và giá trị của các đánh giái đó:
def get_items_rated_by_user(rate_matrix, user_id):
    """
    return (item_ids, scores)
    """
    y = rate_matrix[:,0] # all users
    # item indices rated by user_id
    # we need to +1 to user_id since in the rate_matrix, id starts from 1
    # but id in python starts from 0
    ids = np.where(y == user_id +1)[0]
    item_ids = rate_matrix[ids, 1] - 1 # index starts from 0
    scores = rate_matrix[ids, 2]
    return (item_ids, scores)
#Bây giờ, ta có thể đi tìm vector trọng số của mỗi người dùng:
from sklearn.linear_model import Ridge
from sklearn import linear_model
d = X.shape[1] # data dimension
W = np.zeros((d, n_users))
b = np.zeros(n_users)
for n in range(n_users):
    ids, scores = get_items_rated_by_user(rate_train, n)
    model = Ridge(alpha=0.01, fit_intercept = True)
    Xhat = X[ids, :]
    model.fit(Xhat, scores)
    W[:, n] = model.coef_
    b[n] = model.intercept_
#Sau khi tính được các hệ số W và b, mức độ quan tâm của mỗi người dùng tới một bộ phim được dự đoán bởi:
Yhat = X.dot(W) + b
#Dưới đây là một ví dụ với người dùng có id bằng 10:
n = 10
np.set_printoptions(precision=2) # 2 digits after .
ids, scores = get_items_rated_by_user(rate_test, n)
print('Rated movies ids :', ids )
print('True ratings :', scores)
print('Predicted ratings:', Yhat[ids, n])
#Để đánh giá mô hình tìm được, chúng ta sẽ sử dụng căn bậc hai sai số trung bình bình phường
#(root mean squared error, RMSE)
def evaluate(Yhat, rates, W, b):
    se = cnt = 0
    for n in xrange(n_users):
        ids, scores_truth = get_items_rated_by_user(rates, n)
        scores_pred = Yhat[ids, n]
        e = scores_truth - scores_pred
        se += (e*e).sum(axis = 0)
        cnt += e.size
        return np.sqrt(se/cnt)
print('RMSE for training: %.2f' %evaluate(Yhat, rate_train, W, b))
print('RMSE for test : %.2f' %evaluate(Yhat, rate_test, W, b))