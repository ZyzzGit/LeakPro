from ts2vec import TS2Vec
from torch import cuda, is_tensor, os

def train_ts2vec(train_data, num_variables, batch_size=256):

    # TS2Vec expects a numpy object
    if is_tensor(train_data):
        train_data = train_data.numpy()

    device = "cuda:0" if cuda.is_available() else "cpu"

    model = TS2Vec(
        input_dims=num_variables,
        device=device,
        batch_size=batch_size
    )

    model.fit(
        train_data
    )
    
    if not os.path.exists('data'):
        os.makedirs('data')
    model.save(f'data/ts2vec_model.pkl')