from gwpy.timeseries import TimeSeries, TimeSeriesDict
from pathlib import Path
import requests
from gwpy.segments import Segment, SegmentList
from functools import reduce
import operator
from tqdm import tqdm

def fetch_segments(base_url: str, detector: str='H1'):
    timeline = f'{detector}_DATA'
    url = f'{base_url}/{timeline}/segments'
    
    response = requests.get(url)
    response.raise_for_status()
    
    data = response.json()
    
    segments = []
    for seg in data['results']:
        start = int(seg['start'])
        end = int(seg['stop'])
        segments.append(Segment(start, end))

    return SegmentList(segments)

def load_data(base_url: str, ifos: list[str], sample_rate: int, data_dir: str):

    background_dir = data_dir
    background_dir.mkdir(parents=True, exist_ok=True)

    segments = {}
    for ifo in ifos:
        segments[ifo] = fetch_segments(base_url, ifo)
    network_segments = reduce(operator.and_, segments.values())

    for (start, end) in tqdm(network_segments):
        duration = end - start
        fname = background_dir / f'background-{start}-{duration}.hdf5'
        if fname.exists():
            continue

        ts_dict = TimeSeriesDict()
        for ifo in ifos:
            ts_dict[ifo] = TimeSeries.fetch_open_data(ifo, start, end, cache=False)
        ts_dict = ts_dict.resample(sample_rate)
        ts_dict.write(fname, format='hdf5')

if __name__ == '__main__':
    obs = 'O3a'
    base_url = f'https://gwosc.org/api/v2/runs/{obs}/timelines'
    data_dir = Path(f'/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/{obs}_H1_L1_V1_4096Hz')
    load_data(base_url=base_url, ifos=['H1', 'L1', 'V1'], sample_rate=4096, data_dir=data_dir)