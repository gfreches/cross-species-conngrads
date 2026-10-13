"""Interactive viewer for the human-chimpanzee temporal lobe gradients.

Reads the gradients stored in this repository (no pipeline run needed) and serves three tabs:

  Individual gradients     single-species gradients on the left and right surfaces of either species
  Cross-species gradients  the joint gradients on the surfaces of both species
  Interactive explorer     all profiles in gradient space; click a point to see its connectivity profile, its
                           location on the surface and its closest neighbour (in the other or the same species)

Usage (from anywhere, after pip install -r requirements.txt):
    python code/app.py                 # then open http://127.0.0.1:8050
    python code/app.py --port 8080 --host 0.0.0.0

For a production server, point a WSGI server at the Flask app, e.g. gunicorn --chdir code app:server
"""
import argparse

import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import common as c

COLORS = {('human', 'L'): 'blue', ('human', 'R'): 'cornflowerblue',
          ('chimpanzee', 'L'): 'red', ('chimpanzee', 'R'): 'lightcoral'}
COLORSCALES = ['RdBu_r', 'Spectral_r', 'Viridis', 'Plasma', 'Turbo']
KEYS = list(COLORS)
SOURCES = {'cross_species': 'Cross-species', 'human': 'Human', 'chimpanzee': 'Chimpanzee'}


# ---------- data (loaded once, at start-up) ----------

def full_surface(key, values):
    """Temporal lobe values (vertices x gradients) placed on the whole surface (NaN elsewhere)."""
    out = np.full((len(SURFACE[key][0]), values.shape[1]), np.nan)
    out[MASK[key]] = values
    return out


def explorer_table(maps, keys):
    """One row per temporal lobe vertex with its species, hemisphere, surface vertex and gradient values."""
    parts = [pd.DataFrame(maps[k], columns=[f'g{i + 1}' for i in range(maps[k].shape[1])])
             .assign(species=k[0], hem=k[1], vertex=MASK[k]) for k in keys]
    return pd.concat(parts, ignore_index=True)


SURFACE = {k: c.surface(c.ROOT, *k) for k in KEYS}
MASK = {k: c.mask_indices(c.ROOT, *k) for k in KEYS}
PROFILES = {k: c.profiles(c.ROOT, *k) for k in KEYS}
single = c.single_species(c.ROOT)
cross = c.cross_species(c.ROOT)[1]
MAPS = {'cross_species': {k: full_surface(k, cross[k]) for k in KEYS},
        **{sp: {k: full_surface(k, single[k]) for k in KEYS if k[0] == sp} for sp in c.SPECIES}}
TABLES = {'cross_species': explorer_table(cross, KEYS),
          **{sp: explorer_table(single, [k for k in KEYS if k[0] == sp]) for sp in c.SPECIES}}
N_GRADIENTS = c.N_GRADIENTS


def value_range(maps, g):
    """Shared colour range of gradient g across all surfaces of a data source."""
    values = np.concatenate([m[:, g] for m in maps.values()])
    return float(np.nanmin(values)), float(np.nanmax(values))


def profile(key, vertex):
    return PROFILES[key][np.searchsorted(MASK[key], vertex)]


# ---------- figures ----------

def empty(title='', height=350):
    return go.Figure(layout={'title_text': title, 'height': height,
                             'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)'})


def surface_figure(key, values, title, colorscale, crange, colorbar=True, height=450, marker=None):
    """Lateral view of the inflated surface: temporal lobe coloured by values, the rest grey, optional marker."""
    xyz, faces = SURFACE[key]
    in_tl = np.isfinite(values)[faces].all(1)
    light = dict(lighting=dict(ambient=0.65, diffuse=0.7, specular=0.1), lightposition=dict(x=100, y=200, z=300))
    fig = go.Figure()
    for tl in (False, True):
        f = faces[in_tl == tl]
        used = np.unique(f)
        remap = np.zeros(len(xyz), int)
        remap[used] = np.arange(len(used))
        colour = dict(intensity=values[used], intensitymode='vertex', colorscale=colorscale, cmin=crange[0],
                      cmax=crange[1], showscale=colorbar, colorbar=dict(title='Value', len=0.6)) if tl else \
            dict(color='rgb(211,211,211)')
        fig.add_trace(go.Mesh3d(x=xyz[used, 0], y=xyz[used, 1], z=xyz[used, 2], i=remap[f[:, 0]], j=remap[f[:, 1]],
                                k=remap[f[:, 2]], hoverinfo='none', **colour, **light))
    if marker is not None:
        x, y, z = xyz[marker]
        fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode='markers', hoverinfo='skip',
                                   marker=dict(size=8, color='yellow', line=dict(width=2, color='black'))))
    axis = dict(visible=False)
    fig.update_layout(title_text=title, title_font_size=12, height=height, margin=dict(l=5, r=60, t=40, b=5),
                      scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode='data',
                                 camera=dict(eye=dict(x=-1.7 if key[1] == 'L' else 1.7, y=0, z=0))))
    return fig


def spider(values, title, colour):
    """Connectivity profile over the 20 tracts."""
    fig = go.Figure(go.Scatterpolar(r=np.r_[values, values[:1]], theta=c.TRACTS + c.TRACTS[:1], fill='toself',
                                    line_color=colour))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, max(values.max() * 1.1, 0.05)]),
                                 angularaxis=dict(tickfont=dict(size=8))),
                      showlegend=False, title_text=title, title_font_size=12, height=400,
                      margin=dict(l=30, r=30, t=50, b=30))
    return fig


def scatter(df, x, y, selected=None, closest=None, xrange=None, yrange=None):
    """Profiles in gradient space, with marginal histograms; the selected point and its closest neighbour marked."""
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True, column_widths=[0.92, 0.08],
                        row_heights=[0.08, 0.92], horizontal_spacing=0.01, vertical_spacing=0.01,
                        specs=[[{'type': 'histogram'}, {}], [{'type': 'scattergl'}, {'type': 'histogram'}]])
    hover = (f'{x}: %{{x:.4f}}<br>{y}: %{{y:.4f}}<br>species: %{{customdata[1]}}<br>hem: %{{customdata[2]}}<br>'
             'vertex: %{customdata[3]}<extra></extra>')
    for (sp, h), grp in df.groupby(['species', 'hem'], sort=False):
        colour = COLORS[(sp, h)]
        fig.add_trace(go.Histogram(x=grp[x], marker_color=colour, opacity=0.8, showlegend=False), row=1, col=1)
        fig.add_trace(go.Histogram(y=grp[y], marker_color=colour, opacity=0.8, showlegend=False), row=2, col=2)
        fig.add_trace(go.Scattergl(x=grp[x], y=grp[y], mode='markers', name=f'{sp.capitalize()} {h}',
                                   marker=dict(size=8, opacity=0.88, color=colour, line=dict(width=1, color='black')),
                                   customdata=np.column_stack([grp.index, grp.species, grp.hem, grp.vertex]),
                                   hovertemplate=hover), row=2, col=1)
    for idx, style in ((selected, dict(size=19, color='black', line=dict(width=4, color='yellow'), symbol='x')),
                       (closest, dict(size=30, line=dict(width=6, color='yellow'), symbol='star'))):
        if idx is not None:
            r = df.loc[idx]
            style.setdefault('color', COLORS[(r.species, r.hem)])
            fig.add_trace(go.Scatter(x=[r[x]], y=[r[y]], mode='markers', marker=style, showlegend=False,
                                     customdata=[[idx, r.species, r.hem, r.vertex]], hovertemplate=hover),
                          row=2, col=1)
    xl, yl = f'Gradient {x[1:]}', f'Gradient {y[1:]}'
    fig.update_layout(height=950, title_text=f'Temporal lobe embedding ({xl} vs {yl})', uirevision=True,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.update_xaxes(title_text=xl, row=2, col=1, range=xrange)
    fig.update_yaxes(title_text=yl, row=2, col=1, range=yrange)
    return fig


# ---------- layout ----------

def dropdown(label, name, options, value, width=160):
    return html.Div([html.Label(label, style={'fontWeight': 'bold', 'marginRight': '8px'}),
                     dcc.Dropdown(id=name, options=options, value=value, clearable=False, style={'width': f'{width}px'})],
                    style={'display': 'flex', 'alignItems': 'center'})


GRADIENTS = [{'label': f'Gradient {i + 1}', 'value': i} for i in range(N_GRADIENTS)]
AXES = [{'label': f'Gradient {i + 1}', 'value': f'g{i + 1}'} for i in range(N_GRADIENTS)]
BAR = {'display': 'flex', 'gap': '24px', 'padding': '16px', 'flexWrap': 'wrap', 'alignItems': 'center'}

tab_individual = html.Div([
    html.Div([dropdown('Species:', 'tab1-species', [{'label': s.capitalize(), 'value': s} for s in c.SPECIES], 'human', 220),
              dropdown('Gradient:', 'tab1-gradient', GRADIENTS, 0),
              dropdown('Colorscale:', 'tab1-colorscale', COLORSCALES, 'RdBu_r', 200)], style=BAR),
    html.Div([dcc.Graph(id='tab1-L', style={'flex': '1'}), dcc.Graph(id='tab1-R', style={'flex': '1'})],
             style={'display': 'flex', 'gap': '10px'})])

tab_cross = html.Div([
    html.Div([dropdown('Gradient:', 'tab2-gradient', GRADIENTS, 0),
              dropdown('Colorscale:', 'tab2-colorscale', COLORSCALES, 'RdBu_r', 200)], style=BAR),
    html.Div(id='tab2-surfaces')])


def detail(title, prefix):
    return html.Div([html.H4(title, style={'margin': '0 0 4px 0'}),
                     html.Div([dcc.Graph(id=f'{prefix}-spider', figure=empty(height=400), style={'width': '48%'}),
                               dcc.Graph(id=f'{prefix}-surface', figure=empty(), style={'width': '48%'})],
                              style={'display': 'flex', 'justifyContent': 'space-between'})])


tab_explorer = html.Div([
    dcc.Store(id='zoom'), dcc.Store(id='selected'),
    html.Div([dropdown('Data source:', 'source', [{'label': v, 'value': k} for k, v in SOURCES.items()], 'cross_species', 200),
              dropdown('X axis:', 'x-axis', AXES, 'g1'), dropdown('Y axis:', 'y-axis', AXES, 'g2'),
              dropdown('Show on surface:', 'surface-gradient', AXES, 'g1'),
              dropdown('Colorscale:', 'explorer-colorscale', COLORSCALES, 'RdBu_r', 200)],
             style={**BAR, 'borderBottom': '1px solid #ddd', 'marginBottom': '8px'}),
    html.Div([
        dcc.Graph(id='scatter', figure=empty('Loading...', 950)),
        html.Div([
            detail('Clicked point', 'clicked'),
            html.Div([dropdown('Distance:', 'distance', [{'label': 'Euclidean', 'value': 'euclidean'},
                                                         {'label': 'X axis only', 'value': 'x'},
                                                         {'label': 'Y axis only', 'value': 'y'}], 'euclidean'),
                      dropdown('Match:', 'match', [{'label': 'Other species', 'value': 'different'},
                                                   {'label': 'Same species', 'value': 'same'}], 'different')],
                     style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px', 'padding': '12px 0'}),
            detail('Closest neighbour', 'closest')], style={'display': 'flex', 'flexDirection': 'column', 'gap': '12px'})],
        style={'display': 'grid', 'gridTemplateColumns': 'minmax(700px, 1fr) 850px', 'gap': '20px', 'width': '100%'})])

app = dash.Dash(__name__, title='Cross-species connectivity gradients',
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}])
server = app.server
app.layout = html.Div([
    html.H2('Cross-Species Connectivity Gradients', style={'padding': '0 16px'}),
    dcc.Tabs(value='explorer', children=[dcc.Tab(tab_individual, label='Individual Gradients', value='individual'),
                                         dcc.Tab(tab_cross, label='Cross-Species Gradients', value='cross'),
                                         dcc.Tab(tab_explorer, label='Interactive Explorer', value='explorer')]),
    html.Footer(html.A('GitHub: gfreches/cross-species-conngrads', href='https://github.com/gfreches/cross-species-conngrads',
                       target='_blank'),
                style={'textAlign': 'center', 'padding': '20px 16px', 'marginTop': '24px',
                       'borderTop': '1px solid #ddd', 'color': '#666', 'fontSize': '14px'})])


# ---------- callbacks ----------

@app.callback(Output('tab1-L', 'figure'), Output('tab1-R', 'figure'),
              Input('tab1-species', 'value'), Input('tab1-gradient', 'value'), Input('tab1-colorscale', 'value'))
def individual_surfaces(species, g, colorscale):
    maps = MAPS[species]
    crange = value_range(maps, g)
    return [surface_figure((species, h), maps[(species, h)][:, g], f'{species.capitalize()} {h} - Gradient {g + 1}',
                           colorscale, crange, height=500) for h in c.HEMIS]


@app.callback(Output('tab2-surfaces', 'children'), Input('tab2-gradient', 'value'), Input('tab2-colorscale', 'value'))
def cross_surfaces(g, colorscale):
    maps = MAPS['cross_species']
    crange = value_range(maps, g)
    rows = []
    for sp in c.SPECIES:
        graphs = [dcc.Graph(figure=surface_figure((sp, h), maps[(sp, h)][:, g], f'{sp.capitalize()} {h}', colorscale,
                                                  crange, colorbar=h == 'R'), style={'flex': '1'}) for h in c.HEMIS]
        rows.append(html.Div([html.H4(sp.capitalize(), style={'margin': '8px 0 4px 16px'}),
                              html.Div(graphs, style={'display': 'flex', 'gap': '10px'})]))
    return html.Div(rows, style={'display': 'flex', 'flexDirection': 'column', 'gap': '16px'})


@app.callback(Output('match', 'value'), Input('source', 'value'), prevent_initial_call=True)
def default_match(source):
    return 'different' if source == 'cross_species' else 'same'


@app.callback(Output('zoom', 'data'), Input('scatter', 'relayoutData'), State('zoom', 'data'), prevent_initial_call=True)
def keep_zoom(relayout, zoom):
    zoom = zoom or {}
    for axis in ('xaxis', 'yaxis'):
        if f'{axis}.range[0]' in relayout:
            zoom[axis] = [relayout[f'{axis}.range[0]'], relayout[f'{axis}.range[1]']]
        elif f'{axis}.autorange' in relayout:
            zoom[axis] = None
    return zoom


@app.callback(Output('selected', 'data'), Output('clicked-spider', 'figure'), Output('closest-spider', 'figure'),
              Output('clicked-surface', 'figure'), Output('closest-surface', 'figure'), Output('scatter', 'figure'),
              Input('scatter', 'clickData'), Input('distance', 'value'), Input('match', 'value'),
              Input('x-axis', 'value'), Input('y-axis', 'value'), Input('source', 'value'),
              Input('surface-gradient', 'value'), Input('explorer-colorscale', 'value'),
              State('zoom', 'data'), State('selected', 'data'))
def explore(click, distance, match, x, y, source, surface_g, colorscale, zoom, selected):
    df, zoom = TABLES[source], zoom or {}
    trigger = callback_context.triggered[0]['prop_id'].split('.')[0] if callback_context.triggered else ''
    if trigger in ('', 'x-axis', 'y-axis', 'source'):
        selected = None
    elif trigger == 'scatter' and click:
        idx = int(click['points'][0]['customdata'][0])
        selected = None if idx == selected else idx
    if selected is None or selected not in df.index:
        return None, empty(height=400), empty(height=400), empty(), empty(), \
            scatter(df, x, y, xrange=zoom.get('xaxis'), yrange=zoom.get('yaxis'))

    g = int(surface_g[1:]) - 1
    maps = MAPS[source]
    crange = value_range(maps, g)

    def details(idx, prefix):
        r = df.loc[idx]
        key, label = (r.species, r.hem), f'{prefix}{r.species.capitalize()} {r.hem} (vertex {r.vertex})'
        return (spider(profile(key, r.vertex), label, COLORS[key]),
                surface_figure(key, maps[key][:, g], label, colorscale, crange, height=350, marker=r.vertex))

    s = df.loc[selected]
    candidates = df[df.species != s.species] if match == 'different' else df[(df.species == s.species) & (df.index != selected)]
    d = {'euclidean': np.hypot(candidates[x] - s[x], candidates[y] - s[y]),
         'x': (candidates[x] - s[x]).abs(), 'y': (candidates[y] - s[y]).abs()}[distance]
    closest = int(d.idxmin())
    (s_spider, s_surface), (c_spider, c_surface) = details(selected, ''), details(closest, 'Closest: ')
    return selected, s_spider, c_spider, s_surface, c_surface, \
        scatter(df, x, y, selected, closest, zoom.get('xaxis'), zoom.get('yaxis'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8050)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
