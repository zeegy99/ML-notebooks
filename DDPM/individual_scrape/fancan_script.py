import yt_dlp
import os
'''
Script that finds videos of Kpop idols. This probably was very unnecessary as you can 
manually find these videos buuuuuuuut.  
'''
import sys 

sys.stdout.reconfigure(encoding="utf-8")

def find_m2_fancams(query="원희", aliases=("원희", "Wonhee"), max_videos=50):
    url = f"https://www.youtube.com/@MnetM2/search?query={query}"

    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "ignoreerrors": True,
        "playlistend": max_videos,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    urls = []
    for entry in info.get("entries", []):
        if not entry:
            continue
        title = entry.get("title") or ""
        if any(a.lower() in title.lower() for a in aliases):
            urls.append((title, f"https://www.youtube.com/watch?v={entry['id']}"))
    return urls


if __name__ == "__main__":
    for title, u in find_m2_fancams("원희 Fancam"):
        print(u, "|", title)