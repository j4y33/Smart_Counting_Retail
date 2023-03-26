from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import umap
import matplotlib.pyplot as plt
import numpy as np

from bokeh.plotting import figure, output_file, save
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Inferno256
from io import BytesIO
from PIL import Image
import base64
import cv2
from typing import Union


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='Path to crops folder')
    parser.add_argument('--output', type=str, required=False, default='out.html',
                        help='Path to save html plot')
    parser.add_argument('--w', type=int, required=False, default=1600,
                        help='output width')
    parser.add_argument('--h', type=int, required=False, default=900,
                        help='output heigth')
    parser.add_argument('--point_size', type=int, required=False, default=10,
                        help='point size')
    return parser.parse_args()


def embeddable_image(image_path: Union[str, Path]) -> str:
    """
    get image to show in html on hover
    """
    img_data = cv2.imread(str(image_path))
    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img_data).resize((64, 128), Image.BICUBIC)
    buffer = BytesIO()
    image.save(buffer, format='png')
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()


def get_all_embeddings(folder: Path) -> pd.DataFrame:
    """
    load embeddings from crops
    """
    all_embeddings = []
    for track in folder.glob('*'):
        embeddings_file = track / 'embeddings.json'
        if embeddings_file.exists():
            track_embeddings = pd.read_json(embeddings_file, lines=True)
            track_embeddings['track'] = track.stem
            track_embeddings['image'] = track/(track_embeddings['image_id']+'.jpg')
            all_embeddings.append(track_embeddings)
    embeddings = pd.concat(all_embeddings, ignore_index=True)
    return embeddings


def get_reduced_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    get embeddings reduced to 2-dim with umap
    """
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(df.embedding.tolist())
    df_reduced = pd.DataFrame(embedding, columns=('x', 'y'))
    df_reduced['track'] = [str(x) for x in df.track]
    df_reduced['image'] = list(map(embeddable_image, df.image))
    return df_reduced


def get_colors(df: pd.DataFrame) -> (str):
    """
    get colors for tracks
    """
    palletts_num = len([x for x in df.track.unique()])//256+1
    color_mapping = CategoricalColorMapper(factors=[x for x in df.track.unique()],
                                           palette=Inferno256*palletts_num)
    return color_mapping


def save_plot_with_embeddings(datasource: ColumnDataSource,
                              color_mapping: (str),
                              file: str,
                              width: int,
                              height: int,
                              point_size: int):
    """
    save plot as interactive html file
    """
    plot_figure = figure(
        title='UMAP projection of the Digits dataset',
        plot_width=width,
        plot_height=height,
        tools=('pan, wheel_zoom, reset')
    )

    plot_figure.add_tools(HoverTool(tooltips="""
    <div>
        <div>
            <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
        </div>
        <div>
            <span style='font-size: 16px'>@track</span>
        </div>
    </div>
    """))

    plot_figure.circle(
        'x',
        'y',
        source=datasource,
        color=dict(field='track', transform=color_mapping),
        line_alpha=0.6,
        fill_alpha=0.6,
        size=point_size
    )
    output_file(file)
    save(plot_figure)


if __name__ == '__main__':
    args = get_args()
    folder = Path(args.folder)
    embeddings = get_all_embeddings(folder)
    datasource = ColumnDataSource(get_reduced_df(embeddings))
    color_mapping = get_colors(embeddings)
    save_plot_with_embeddings(datasource, color_mapping, args.output, args.w, args.h, args.point_size)
