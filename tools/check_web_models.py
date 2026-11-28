"""Smoke-test a running LeukoGuard deployment using only illustrative values."""
import argparse
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def check(base_url):
    def request(path, data=None):
        body = None if data is None else json.dumps(data).encode()
        req = Request(base_url.rstrip("/") + path, data=body,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=90) as response:
            return response.read()

    health = json.loads(request("/api/health"))
    assert health["status"] == "ok" and len(health["models"]) == 4
    routine = json.loads(request("/api/predict", {"wbc": 7, "hgb": 13.5, "plt": 250}))
    assert 0 <= routine["score"] <= 1 and isinstance(routine["flagged"], bool)
    assert b"panel.html" in request("/") and b"panel.js" in request("/panel.html")

    schema = json.loads(request("/api/analyzer-panel"))
    expected = ["Logistic Regression", "Random Forest", "XGBoost"]
    assert schema["models"] == expected
    example = json.loads(request("/panel-example.json"))
    assert set(example) == {field["name"] for group in schema["groups"].values() for field in group}
    panel = json.loads(request("/api/analyzer-panel/predict", example))
    assert panel["n_entered"] == panel["n_features"] == 71
    assert [r["model"] for r in panel["results"]] == expected
    for result in panel["results"]:
        assert 0 <= result["score"] <= 1 and 0 < result["threshold"] < 1
        assert result["flagged"] == (result["score"] >= result["threshold"])

    missing = {**example, "MO-WY": None}
    partial = json.loads(request("/api/analyzer-panel/predict", missing))
    assert partial["n_entered"] == 70 and len(partial["results"]) == 3
    for invalid in [{}, {"WBC": 7}, {**example, "unknown": 1}, {**example, "WBC": -1}]:
        try:
            request("/api/analyzer-panel/predict", invalid)
        except HTTPError as exc:
            assert exc.code == 422
        else:
            raise AssertionError("The analyzer endpoint accepted invalid input")
    print("Passed routine prediction, all three analyzer predictions, missing values and input validation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", nargs="?", default="http://127.0.0.1:8000")
    check(parser.parse_args().base_url)
