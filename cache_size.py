import fastf1

fastf1.Cache.enable_cache('cache')

# Get cache info
path, size = fastf1.Cache.get_cache_info()

if path:
    print(f"Cache location: {path}")
    print(f"Cache size: {size / (1024**3):.2f} GB")
else:
    print("Cache not configured")

# Print cache representation
print(fastf1.Cache)
# Output: FastF1 cache (1.23 GB) /path/to/cache