from sklearn.datasets import load_wine
import pandera as pa
import pandera.typing as pat

DatasetWine = pat.DataFrame['ModelWine']

class ModelWine(pa.DataFrameModel):
  alcohol: pa.Float64
  malic_acid: pa.Float64
  ash: pa.Float64
  alcalinity_of_ash: pa.Float64
  magnesium: pa.Float64
  total_phenols: pa.Float64
  flavanoids: pa.Float64
  nonflavanoid_phenols: pa.Float64
  proanthocyanins: pa.Float64
  color_intensity: pa.Float64
  hue: pa.Float64
  od280_od315_of_diluted_wines: pa.Float64
  proline: pa.Float64
  target: pa.Int32

  @classmethod
  def load(cls) -> DatasetWine:
    # noinspection PyTypeChecker
    return cls.validate(
      load_wine(as_frame=True)['frame']
      .rename(columns={
        'od280/od315_of_diluted_wines': 'od280_od315_of_diluted_wines',
      }),
    )
