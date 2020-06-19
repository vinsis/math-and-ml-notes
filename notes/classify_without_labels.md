### 1. Use K-means to group representations

Here `group_vectors` are is a numpy array with shape `(num_image, dimension_length)`.

```python
import sklearn
from sklearn.cluster import KMeans

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(group_vectors)
```

Now let's define a function to get nearest neighbors for a given index:

```python
def get_nearest_neighbors(index, num_neighbors=5):
    source_vector = group_vectors[index]
    distances = [np.linalg.norm(source_vector-vector) for vector in group_vectors]
    return np.argsort(distances)[1:1+num_neighbors]
```

Now let's create a model. The loss function is implemented within the model itself.

```python
class Model(nn.Module):
    def __init__(self, output_dims = 2):
        super(Model, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(512,10),
            nn.ReLU(True),
            nn.Linear(10, output_dims)
        )

    def forward(self, x):
        '''
        x is a batch consisting of an image and its nearest neighbors
        '''
        return self.sequential(x).softmax(1)

    def loss(self, output, weight=0.5):
        # output has size (B, num_neighbors, num_classes)
        batch_size = output.size(0)
        total_loss = torch.Tensor([0.0])
        for batch_index in range(batch_size):
            total_loss += self.loss_per_group(output[batch_index], weight)
        return total_loss / batch_size

    def loss_per_group(self, output, weight):
        # output has size (num_neighbors, num_classes)
        batch_size = output.size(0)
        assert batch_size > 1
        source_vector = output[0].unsqueeze(0)
        # dot product
        dot_product = torch.Tensor([0.0])
        for index in range(1, batch_size):
            vector = output[0].unsqueeze(0)
            dot_product += -(source_vector * vector).sum().log()
        # entropy term
        probs = output.mean(0)
        log_probs = probs.log()
        entropy_loss = (probs * log_probs).sum()

        total_loss = dot_product + weight * entropy_loss
        return total_loss
```

The loader looks like this:

```python
class Loader(Dataset):
    def __init__(self, vectors, num_neighbors, num_labels):
        self.vectors = vectors
        self.num_neighbors = num_neighbors
        self.num_vectors = len(vectors)
        self.kmeans = self.perform_kmeans(vectors, num_labels)
        self.labels = self.kmeans.labels_

    def get_nearest_neighbors(self, index, vectors, num_neighbors=None):
        if num_neighbors is None:
            num_neighbors = self.num_neighbors
        source_vector = self.vectors[index]
        distances = [np.linalg.norm(source_vector-vector) for vector in vectors]
        return np.argsort(distances)[1:1+num_neighbors]

    def perform_kmeans(self, vectors, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vectors)
        return kmeans

    def __getitem__(self, index):
        source_vector = self.vectors[index]
        source_label = self.labels[index]
        vectors_with_same_label = self.vectors[self.labels == source_label]
        indices = self.get_nearest_neighbors(index, vectors_with_same_label)
        result = [source_vector]
        for i in indices:
            result.append(self.vectors[i])
        result = np.stack(result)
        return torch.from_numpy(result).float()

    def __len__(self):
        return self.num_vectors
```

Then define the loader object like so:

```python
loader = DataLoader(Loader(group_vectors, num_neighbors=3, num_labels=3), batch_size=16, shuffle=True)
```

Then just train a model:

```python
model = Model(output_dims=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(2):
    for x in loader:
        output = model(x)
        loss = model.loss(output, weight=10)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
```
