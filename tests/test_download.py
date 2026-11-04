"""Two-phase CDS download logic with a fake client (no network)."""
import json

from wfsea.data import download as dl


class FakeRemote:
    def __init__(self, rid, status):
        self.request_id, self.status = rid, status

    def download(self, target):
        open(target, "wb").write(b"CDF-fake")
        return target


class FakeClient:
    def __init__(self):
        self.submitted, self.ready = [], set()

    def submit(self, dataset, req):
        rid = f"id-{req['year'][0]}-{req['month'][0]}"
        self.submitted.append(rid)
        return FakeRemote(rid, "accepted")

    def get_remote(self, rid):
        return FakeRemote(rid, "successful" if rid in self.ready else "running")


def test_submit_once_then_fetch_incrementally(tmp_path, monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(dl, "_client", lambda: fake)
    reqs = dl.pressure_requests([2018], ["z", "t"], [850, 500])
    assert len(reqs) == 12 and reqs["2018-03"]["time"] == ["00:00", "06:00", "12:00", "18:00"]
    state = tmp_path / "cds_requests.json"
    dl.cds_submit(reqs, state, max_new=5)
    dl.cds_submit(reqs, state, max_new=50)                 # second run only submits the remaining 7
    assert len(fake.submitted) == 12 and len(set(fake.submitted)) == 12

    fake.ready = {"id-2018-01", "id-2018-02"}
    st = dl.cds_fetch(state, tmp_path / "out")
    assert st["2018-01"]["status"] == "downloaded" and st["2018-03"]["status"] == "running"
    assert (tmp_path / "out" / "year_2018" / "era5_pressure_2018_01.nc").exists()
    fake.ready |= {"id-2018-03"}
    dl.cds_fetch(state, tmp_path / "out")
    assert json.loads(state.read_text())["2018-03"]["status"] == "downloaded"

    # states from an earlier Kaggle run merge without losing 'downloaded'
    other = tmp_path / "old.json"
    other.write_text(json.dumps({"2018-04": {"request_id": "id-2018-04", "status": "downloaded"}}))
    merged = dl.merge_states([state, other])
    assert merged["2018-04"]["status"] == "downloaded" and merged["2018-01"]["status"] == "downloaded"
