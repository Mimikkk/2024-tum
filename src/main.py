from src.datasets import DatasetIris, DatasetWine

def main():
  iris = DatasetIris.load()
  wine = DatasetWine.load()

if __name__ == '__main__':
  main()
