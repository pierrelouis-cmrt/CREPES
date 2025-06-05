###Fichier permettant de remplir les fichiers albedo_01.csv ... albedo12.csv
#Attention, il faut avoir préablement exécuté construction_csv.py
#Attention, l'exécution de ce programme est tres long (plus d'une heure)
#Ce programme lit les coordonnées des cases du fichier albedo_01.csv, effectue une requette à l'API, et moyenne pour chaque mois. Pour chacun de ces appels, les cases des autres fichiers sont remplis.
#Cette méthode limite le temps d'exécution (une requette pour les 12 mois, plutot que de faire une requette piur chaque mois)
import csv
import requests
import time

def get_power_data(lat, lon, start_date, end_date):
    start_time = time.time()
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_UP&community=RE&longitude={lon}&latitude={lat}&start={start_date}&end={end_date}&format=JSON"
    response = requests.get(url)
    data = response.json()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Temps d'exécution: {execution_time} secondes")
    return data

def calculate_albedo(down_radiation, up_radiation):
    down_values = list(down_radiation.values())
    up_values = list(up_radiation.values())

    average_down = sum(down_values) / len(down_values) if len(down_values) > 0 else 0
    average_up = sum(up_values) / len(up_values) if len(up_values) > 0 else 0

    if average_down == 0:
        albedo = 0
    else:
        albedo = average_up / average_down

    return albedo

def get_monthly_albedo(lat, lon, year):
    start_date = f"{year}0101"
    end_date = f"{year}1231"
    data = get_power_data(lat, lon, start_date, end_date)

    down_radiation = data['properties']['parameter']['ALLSKY_SFC_SW_DWN']
    up_radiation = data['properties']['parameter']['ALLSKY_SFC_SW_UP']

    monthly_albedo = []
    monthly_down_radiation = {i: [] for i in range(1, 13)}
    monthly_up_radiation = {i: [] for i in range(1, 13)}

    for date, value in down_radiation.items():
        month = int(date[4:6])
        monthly_down_radiation[month].append(value)

    for date, value in up_radiation.items():
        month = int(date[4:6])
        monthly_up_radiation[month].append(value)

    for month in range(1, 13):
        down_values = {f"{year}{month:02d}{day:02d}": value for day, value in enumerate(monthly_down_radiation[month], start=1)}
        up_values = {f"{year}{month:02d}{day:02d}": value for day, value in enumerate(monthly_up_radiation[month], start=1)}
        albedo = calculate_albedo(down_values, up_values)
        monthly_albedo.append(albedo)

    return monthly_albedo

def update_albedo_files(year):
    base_filename = "albedo/albedo"
    files = [f"{base_filename}{month:02d}.csv" for month in range(1, 13)]

    # Lire les coordonnées du premier fichier
    with open(files[0], mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)
        latitudes = []
        longitudes = [float(x) for x in header[1:]]
        for row in reader:
            latitudes.append(float(row[0]))

    # Obtenir les albédos pour chaque coordonnée
    for lat in latitudes:
        for lon in longitudes:
            monthly_albedo = get_monthly_albedo(lat, lon, year)

            for month in range(1, 13):
                filename = files[month - 1]
                with open(filename, mode='r') as file:
                    reader = csv.reader(file)
                    rows = list(reader)

                with open(filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(header)
                    for row in rows[1:]:
                        if float(row[0]) == lat:
                            row[header.index(str(lon))] = monthly_albedo[month - 1]
                        writer.writerow(row)

# Appel de la fonction avec l'année spécifique
year = "2023"
update_albedo_files(year)
