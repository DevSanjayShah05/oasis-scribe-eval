from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_and_items():
    assert client.get("/health").json() == {"status": "ok"}
    assert len(client.get("/items").json()) == 18


def test_pdgm_and_qa_endpoints():
    body = {"answers": {"M1800": "2", "M1810": "2", "M1820": "3", "M1830": "5", "M1840": "2",
                        "M1850": "2", "M1860": "4", "M1033": ["1", "3", "4", "7"]},
            "period": {"clinical_group": "MMTA - Cardiac", "timing": "Early",
                       "admission_source": "Institutional", "comorbidity": 1}}
    r = client.post("/pdgm/functional", json=body).json()
    assert r["complete"] and r["functional_points"] == 3 + 5 + 12 + 17 + 6 + 4 + 20 + 12
    assert r["functional_level"] == "High"
    flags = client.post("/qa-check", json=body).json()["flags"]
    assert flags == []  # chairfast + sponge bath with help is consistent


def test_extract_keyword_system_without_llm():
    t = ("NURSE: Let's talk about walking.\nPATIENT: I use my cane, but with it I go anywhere, even the steps outside.\n"
         "NURSE: How are you bathing right now?\nPATIENT: I use the shower chair and the grab bar and I do the whole shower myself.")
    r = client.post("/extract", json={"transcript": t, "system": "keyword"}).json()
    assert r["items"]["M1860"]["value"] == "1"
    assert r["items"]["M1830"]["value"] == "1"
