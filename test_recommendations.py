def test_known_member_returns_200_and_matches_fixture(client, load_sample):
    expected = load_sample("top_picks_200_M-10001.json")
    resp = client.get("/recommendations/M-10001")
    assert resp.status_code == 200
    assert resp.json() == expected


def test_unknown_member_returns_404_and_matches_fixture(client, load_sample):
    expected = load_sample("top_picks_404_M-99999.json")
    resp = client.get("/recommendations/M-99999")
    assert resp.status_code == 404
    assert resp.json() == expected


def test_404_body_has_detail_message(client):
    body = client.get("/recommendations/M-99999").json()
    assert body["detail"] == "Member not found"
