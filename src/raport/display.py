from typing import Any, NamedTuple
from pandas import DataFrame
from IPython.display import display, HTML

def cx(styles: dict[str, Any]): return ';'.join([f'{k}:{v}' for k, v in styles.items()])

def pick_first_column_id(df: DataFrame) -> str:
  return df.columns[0]

def pick_second_column_id(df: DataFrame) -> str:
  return df.columns[1]

def for_template(template, items) -> str:
  return '\n'.join(template(item) for item in items)

TitledFrame = NamedTuple('TitledFrame', [('title', str), ('frame', DataFrame)])

def frame_template(item: TitledFrame) -> str: return f"""
  <div style="{cx({
  'border': 'black solid 1px',
  'border-radius': '0.125rem',
  'display': 'flex',
  'flex-direction': 'column',
  'align-items': 'center',
  'justify-content': 'center',
  'gap': '1rem',
  'padding': '1rem'
})}">
    <span>{item.title}</span>
    {item.frame.to_html()}
  </div>
"""

def frames_template(*dfs: TitledFrame, title: str = None) -> str: return f"""
  <div style="{cx({
  'display': 'flex',
  'flex-direction': 'column',
  'border': 'black solid 1px',
  'border-radius': '0.125rem',
  'overflow': 'auto',
  'align-items': 'center',
  'justify-content': 'center',
  'padding': '1rem',
  'gap': '1rem'
})}">
    {title and f"<bold>{title}</bold>" or ""}
    <div style="{cx({'display': 'flex', 'gap': '1rem'})}">
      {for_template(frame_template, dfs)}
    </div> 
  </div>
"""

def render(template: str):
  display(HTML(template))

def render_frames(
    clean: DataFrame,
    noise: DataFrame,
    noisy: DataFrame,
    column_ids: list[str] = None,
    column_id: str = None,
    title: str = None
):
  if column_id is not None:
    column_ids = [column_id]

  if column_ids is not None:
    clean = clean[column_ids]
    noise = noise[column_ids]
    noisy = noisy[column_ids]

  render(frames_template(
    TitledFrame('Czyste dane', clean),
    TitledFrame('Szum', noise),
    TitledFrame('Zaszumione dane', noisy),
    title=title
  ))

def render_difference(
    clean: DataFrame,
    noisy: DataFrame,
    column_ids: list[str] = None,
    column_id: str = None,
    title: str = None
):
  render_frames(clean, noisy - clean, noisy, column_ids=column_ids, column_id=column_id, title=title)
