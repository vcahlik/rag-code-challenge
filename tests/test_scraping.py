from brainsoft_code_challenge.scraping import scrape_all


def test_scraping():
    results = scrape_all()
    assert len(results) > 0
