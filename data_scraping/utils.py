import requests


def get_openalex_work_title(work_id: str) -> str | None:
    """
    Retrieve the title of an OpenAlex work from its work ID.
    
    Args:
        work_id: The OpenAlex work ID (format: W<number>)
        
    Returns:
        The title of the work, or None if an error occurred
    """
    try:
        url = f"https://api.openalex.org/works/{work_id}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("title")
    except Exception:
        return None


def get_openalex_topic_display_name(topic_id: str) -> str | None:
    """
    Retrieve the display name of an OpenAlex topic from its topic ID.
    
    Args:
        topic_id: The OpenAlex topic ID (format: T<number>)
        
    Returns:
        The display name of the topic, or None if an error occurred
    """
    try:
        url = f"https://api.openalex.org/topics/{topic_id}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("display_name")
    except Exception:
        return None

if __name__ == "__main__":
    # Example usage
    work_id = "W2741809807"
    title = get_openalex_work_title(work_id)
    print(f"Title of work {work_id}: {title}")

    topic_id = "T11636"
    display_name = get_openalex_topic_display_name(topic_id)
    print(f"Display name of topic {topic_id}: {display_name}")