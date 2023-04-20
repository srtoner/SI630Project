import geocoder
import pickle as pkl
import pandas as pd
import time

def get_state(text):
    g = geocoder.geonames(text, key='srtoner')
    c = geocoder.geonames(g.geonames_id, key = 'srtoner', method = 'hierarchy')
    return c.geojson['features']

with open("places.pkl", "rb") as file: # Unique Places
    places = pkl.load(file)

places_unpacked = [item  for item in places.values()]
def unpack_place(place):
    return (place.id, place.name, place.full_name, place.country, place.country_code, place.place_type, place.centroid[0], place.centroid[1])

unpacked_places = [unpack_place(place) for place in places_unpacked]
place_df = pd.DataFrame(unpacked_places, columns = ("id", "name", "full_name", "country", "country_code", "type", "lat", "lon"))
len(place_df)

places = place_df.name
output = []


for p in places:
    try:
        t = get_state(p)
    except:
        time.sleep(3100)
        t = get_state(p)
    output.append(t)

with open('geonames_data.pkl', 'wb') as f:
    pkl.dump(output, f)