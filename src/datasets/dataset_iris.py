from sklearn.datasets import load_iris
import pandera as pa
import pandera.typing as pat

class DatasetIris(pa.DataFrameModel):
  sepal_length: pa.Float64
  sepal_width: pa.Float64
  petal_length: pa.Float64
  petal_width: pa.Float64
  target: pa.Int32

  @classmethod
  def load(cls) -> pat.DataFrame['DatasetIris']:
    # noinspection PyTypeChecker
    return cls.validate(
      load_iris(as_frame=True)['frame']
      .rename(columns={
        'sepal length (cm)': 'sepal_length',
        'sepal width (cm)': 'sepal_width',
        'petal length (cm)': 'petal_length',
        'petal width (cm)': 'petal_width',
      }),
    )
