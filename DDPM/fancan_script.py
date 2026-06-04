import yt_dlp
import os
'''
Script that finds videos of Kpop idols. This probably was very unnecessary as you can 
manually find these videos buuuuuuuut.  
'''


def download_fancams(idol_name, num_results=20, output_dir="videos"):
    #Takes Idol Name, Num Results, Downloads WEBM File to videos

    search_query = f"ytsearch{num_results}:{idol_name} 직캠 fancam 1080p" #Korean Fancam stages
    
    ydl_opts = {
    "outtmpl": f"{output_dir}/%(title)s.%(ext)s",
    "match_filter": yt_dlp.utils.match_filter_func("duration < 600 & duration > 60"),
    "ignoreerrors": True,
    "quiet": False,
    "extractor_args": {
        "youtube": {"player_client": ["tv_embedded"]}
    },
    "format": "bestvideo[height>=1080]/bestvideo",
    "verbose": True,
}
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([search_query])
        


if __name__ == "__main__":
    download_fancams("Wonhee", num_results=1)