import os
import pandas as pd


def find_src_path(android_dir):
    for root, _, _ in os.walk(android_dir):
        if (root.endswith('src')):
            return root


def find_files_recursively(src_dir: str):
    """find all .java files recursively from src dir"""
    java_files = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.java') or file.endswith('.kt'):
                java_files.append(root + '/' + file)
    return java_files


def get_java_paths(base) -> pd.DataFrame:
    """read directories and find paths of all 'src' directories"""
    all_dirs = []
    all_java_paths = []
    for android_dir in os.listdir(base):
        
        if android_dir == 'AdAway-master':
          path = base / android_dir / 'app'
        elif android_dir == 'android-template-master':
          path = base / android_dir / 'template'
        elif android_dir == 'zanavi-master':
          path = base / android_dir / 'navit' /'android'
        else:
          path = base / android_dir
        src = find_src_path(path)
        if src:
            java_paths = find_files_recursively(src)
            all_dirs.extend(len(java_paths) * [android_dir])
            all_java_paths.extend(java_paths)
    df = pd.DataFrame({"projects": all_dirs, "java_paths": all_java_paths})
    return df
