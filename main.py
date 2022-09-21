import city as ct
import matplotlib.pyplot as plt
import geopandas
import pandas as pd
import numpy as np
from shapely.geometry import Point
import contextily as cx
import folium
import matplotlib.animation as animation

CITY = 'Львів'
POPULATION = 10000
PLANTS = 1000
TRANSPORT = .2
RESTAURANT = 0.2
VACCINATION = (False, .75, 0.8) #Acctivate, %of peoples, %immunity
HOUSES = POPULATION // 3
PROBA_INFECTION = 0.005
PROBA_DEATH = .1
SICKNESS_DAY = 30
STATES = ["S", "I", "R", "D"]
sim_stat = pd.DataFrame(columns=STATES + ['Rn', 'R0'])
col_p = {}
col_p['S'] = 'green'
col_p['I'] = 'red'
col_p['R'] = 'grey'
col_p['D'] = 'black'
ISOLATION = (True, 1, 14, 11) # (Acctivate, probability, incubation time, start)
WITHOUT_SYMPTOM = 0.01
SECURITY = (False, 50, 0.3) #security of person
SECOND_WAVE = (True, 300) #Activation, days
CONTAGIOUSNESS = (15, 3, 9)

# waves
# PROBA_INFECTION = .2
# PLANTS = 1000
# ISOLATION = (True, 1, 1, 100) # (Acctivate, probability, incubation time, start)
# WITHOUT_SYMPTOM = 0
#Waves2
# PLANTS = 100
# PROBA_INFECTION = .01
# ISOLATION = (True, 1, 2, 100) # (Acctivate, probability, incubation time, start)
# ISOLATION = (True, 1, 3, 10)
# WITHOUT_SYMPTOM = 0.2



gdf = geopandas.read_file('official_ukraine_administrative_boundary_shapefile-master/SETTLEMENT.shp')
lv = gdf[gdf['NAME_UA_se'] == CITY]
lv = lv[lv.index == 14183]
# print(lv.crs)
# exit()
city = ct.city(lv['geometry'].values[0], POPULATION, PLANTS, HOUSES, PROBA_INFECTION,
               SICKNESS_DAY, PROBA_DEATH, ISOLATION, WITHOUT_SYMPTOM, TRANSPORT, VACCINATION, SECURITY, SECOND_WAVE,
               CONTAGIOUSNESS, RESTAURANT, file=True)
city.population.loc[0, 'State'] = "I"
# city.population.loc[44, 'State'] = "I"
# city.population.loc[2, 'State'] = "I"

def calc_stat():
    global sim_stat, city
    sim_res = pd.DataFrame(city.population.State.value_counts()).transpose()
    try:
        # print(sim_stat.iloc[-1])
        I0 = sim_stat.iloc[-2]["I"]
        I1 = sim_res.iloc[-1]["I"]
        S = sim_res.iloc[-1]["S"]/sim_res.iloc[-1].sum()
        R = I1/I0
        R0 = I1 / (I0 * S)
        # x = sim_res.iloc[0]["I"]/sim_res.iloc[0].sum()
        # sec = 1/(1+np.exp(-50*(x-0.1)))
    except:
        R = np.NaN
        R0 = np.NaN
        # sec = np.NaN
    sim_res['Rn'] = R
    sim_res['R0'] = R0
    # sim_res['seq'] = sec
    sim_stat = pd.concat([sim_stat, sim_res], ignore_index=True)

def plot_init(ax1, plt_city, title=""):
    calc_stat()
    lv["geometry"].plot(alpha=.2, ax=ax1)
    cx.add_basemap(ax1, crs=lv.crs, alpha=0.8, attribution="")
    for st, in ["D", "R", "S", "I"]:
        group = pd.unique(city.population[city.population.State == st].House)
        points = city.houses[city.houses.index.isin(group)].geometry.to_crs('EPSG:4326')
        plt_city[st],  = ax1.plot(points.x, points.y, 'o', color=col_p[st], linestyle='', ms=3)

    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.tick_params(
        left=False,
        bottom=False,  # ticks along the bottom edge are off
        labelbottom=False)
    ax1.set_title(title, fontsize=8)
    ttl = ax1.title
    ttl.set_position([0.5, 0.95])

    # extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # fig.savefig('ax1_figure_expanded.png', bbox_inches=extent.expanded(.5, 1.2),dpi=600)

    # fig.savefig("test")

def generate_day():
    while True:
        city.move(len(sim_stat))
        yield city

def update(city):
    if (len(sim_stat)-1) % 30 == 0:
        extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig('Lviv' + str(len(sim_stat)), bbox_inches=extent.expanded(1.1, 1.2),dpi=600)

    calc_stat()
    ax2.clear()
    sim_stat[sim_stat.columns[:-2]].plot(ax=ax2, color=list(col_p.values()))
    for st, in ["D", "R", "S", "I"]:
        group = pd.unique(city.population[city.population.State == st].House)
        points = city.houses[city.houses.index.isin(group)].geometry.to_crs('EPSG:4326')
        plt_city[st].set_data(points.x, points.y)

    #
    # for st, in STATES:
    #     group = city.peoples[city.peoples.state == st]
    #     plt_city[st].set_data(group.x, group.y)

    if np.isnan(sim_stat["I"].iloc[-1]):
        ani.event_source.stop()



def without_animation():
    for _ in generate_day():
        calc_stat()
        if np.isnan(sim_stat["I"].iloc[-1]) or len(sim_stat)>=1900:
            break

def with_animation():
    global ax1, ax2, ani, plt_city, plt_stat, fig
    fig = plt.figure()
    plt_city = dict()
    plt_stat = dict()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    plot_init(ax1, plt_city, "Lviv")  # ініціалізація міста


    ani = animation.FuncAnimation(fig, update, generate_day, interval=1, repeat=True, save_count=300)
    plt.show()
    # ani.save('coil.gif', writer=animation.PillowWriter(fps=30))


# def contagiousness(day, c, a, b):
#     day = day
#     if day<=c:
#         x = (c - day)/a
#         res = max(0, ((1 - x**2)**0.5).real)
#     else:
#         x = (day - c) / b
#         res = np.exp(-abs(x)**3)
#     return res


import time
s_time = time.time()

# import matplotlib.pyplot as plt
# x = [i for i in range(0,30)]
# y = [contagiousness(i, 15, 3, 9) for i in x]
# plt.plot(x, y)
# plt.show()
# d = pd.DataFrame(y, index=x)
# d.to_csv('contagiousness.csv')
# 
# exit()


# without_animation()
with_animation()
# sim_stat.plot()
# plt.show()
print(sim_stat)
sim_stat.to_csv('sim_stat.csv')
print(f'time: {time.time() - s_time:.5f}s')

city.population[city.population.State == "S"].to_csv("S.csv")

lv["area"] = 0.1
m = lv.explore("area",
               scheme="naturalbreaks",  # use mapclassify's natural breaks scheme
               )
m = city.plants.explore("workers", m=m, cmap = 'Greys', # use red color on all points
                legend=True,
 marker_kwds=dict(radius=10, fill=True), # make marker radius 10px with fill
 tooltip_kwds=dict(labels=False), # do not show column label in the tooltip
    )

m = city.houses.explore("family", m=m, cmap="Reds", # use red color on all points
 marker_kwds=dict(radius=2, fill=True), # make marker radius 10px with fill
 tooltip_kwds=dict(labels=False), # do not show column label in the tooltip
                 )

folium.TileLayer('Stamen Toner', control=True).add_to(m)  # use folium to add alternative tiles
folium.LayerControl().add_to(m)  # use folium to add layer control
m.save('my_map.html')
#
# ax = lv["geometry"].to_crs('EPSG:3857').plot(alpha=.2,)
# city.houses.plot(ax=ax, color='black', alpha=.5, markersize=10)
# city.plants.plot(ax=ax, column='workers', alpha=1 , markersize= 50, legend=True, cmap = 'Greens')
# # cx.add_basemap(ax, crs=lv.crs)
# plt.show()
# plt.savefig("Lviv.jpg")

# def my_geopandas():
#     # import geopandas
#     import webbrowser
#     import os
#     import pandas as pd
#     import json
#     # from simpledbf import Dbf5
#     # dbf = Dbf5('official_ukraine_administrative_boundary_shapefile-master/SETTLEMENT.dbf')
#     # df = dbf.to_dataframe()
#     # print(df)
#
#     gdf = geopandas.read_file('official_ukraine_administrative_boundary_shapefile-master/SETTLEMENT.shp')
#     # print(gdf[gdf['NAME_UA_se'] == 'Львів'])
#     lv = gdf[gdf['NAME_UA_se'] == 'Львів']
#     print(lv[lv.index == 14183])
#     print(lv[lv.index == 14183]["geometry"].bounds)
#     # lv["geometry"].plot()
#     lv[lv.index == 14183]["geometry"].plot()
#     plt.show()
    # print(lv)
    # lv["area"] = lv.area
    # m = lv.explore("area", legend=False)
    # m.save('my_map.html')

    # then open it
    # webbrowser.open('file://' + os.path.realpath('my_map.html'))

    # d = json.load('DA13_3D_Buildings_Merged.city.json')
    # gdf = geopandas.read_file('60199 - Ukraine.vm')
    # for c in gdf.columns:
    #     print(gdf[c])
    # gdf["geometry"].plot()
    # plt.show()


    # path_to_data = geopandas.datasets.get_path("nybb")
    # gdf = geopandas.read_file(path_to_data)
    # gdf = geopandas.read_file('ukraine_geojson-master/Chernivtsi.json')
    # print(gdf)
    #
    # gdf["geometry"].plot()
    # plt.show()


    # gdf.to_file("my_file.geojson", driver="GeoJSON")
    # print(gdf[gdf.columns[:-1]])
    # print(gdf["geometry"])
    # gdf = gdf.set_index("BoroName")
    # gdf["area"] = gdf.area
    # print(gdf["area"])
    # gdf['boundary'] = gdf.boundary
    # print(gdf['boundary'])
    # gdf['centroid'] = gdf.centroid
    # print(gdf['centroid'])
    # first_point = gdf['centroid'].iloc[0]
    # gdf['distance'] = gdf['centroid'].distance(first_point)
    # print(gdf['distance'])
    #
    # gdf.plot("area", legend=True)
    # plt.show()
    #
    # m = gdf.explore("area", legend=False)
    # m.save('my_map.html')

    # then open it
    # webbrowser.open('file://' + os.path.realpath('my_map.html'))

    # plt.show()

    # gdf = gdf.set_geometry("centroid")
    # plt.show()

    # ax = gdf["geometry"].plot()
    # gdf["centroid"].plot(ax=ax, color="black")
    # plt.show()

    # gdf["convex_hull"] = gdf.convex_hull
    # ax = gdf["convex_hull"].plot(alpha=.5)  # saving the first plot as an axis and setting alpha (transparency) to 0.5
    # gdf["boundary"].plot(ax=ax, color="white", linewidth=.5)  # passing the first plot and setting linewitdth to 0.5
    # plt.show()

    # buffering the active geometry by 10 000 feet (geometry is already in feet)
    # gdf["buffered"] = gdf.buffer(10000)
    #
    # buffering the centroid geometry by 10 000 feet (geometry is already in feet)
    # gdf["buffered_centroid"] = gdf["centroid"].buffer(10000)
    #
    # ax = gdf["buffered"].plot(alpha=.5)  # saving the first plot as an axis and setting alpha (transparency) to 0.5
    # gdf["buffered_centroid"].plot(ax=ax, color="red", alpha=.5)  # passing the first plot as an axis to the second
    # gdf["boundary"].plot(ax=ax, color="white", linewidth=.5)  # passing the first plot and setting linewitdth to 0.5
    # plt.show()

    # brooklyn = gdf.loc["Brooklyn", "geometry"]
    # gdf[gdf.index == "Brooklyn"]["geometry"].plot()
    # plt.show()

    # print(type(brooklyn))
    # print(gdf["buffered"].intersects(brooklyn))
    # gdf["within"] = gdf["buffered_centroid"].within(gdf)
    # print(gdf["within"])
    #
    # gdf = gdf.set_geometry("buffered_centroid")
    # ax = gdf.plot("within", legend=True, categorical=True,
    #               legend_kwds={'loc': "upper left"})  # using categorical plot and setting the position of the legend
    # gdf["boundary"].plot(ax=ax, color="black", linewidth=.5)  # passing the first plot and setting linewitdth to 0.5
    # plt.show()

    # print(gdf.crs)
    # gdf = gdf.set_geometry("geometry")
    # boroughs_4326 = gdf.to_crs("EPSG:4326")
    # boroughs_4326.plot()
    # plt.show()
# my_geopandas()

# def my_rand(number, geom):
#     xmin, ymin, xmax, ymax = geom.bounds
#     pts = []
#     for p in range(number):
#         while True:
#             x = (xmax - xmin) * np.random.random() + xmin
#             y = (ymax - ymin) * np.random.random() + ymin
#             if Point(x, y).intersects(geom):
#                 pts.append(Point(x, y))
#                 break
#     pts = geopandas.GeoDataFrame(pts, columns= ['geometry'])
#     return pts

# def city_creation():
#     gdf = geopandas.read_file('official_ukraine_administrative_boundary_shapefile-master/SETTLEMENT.shp')
#     lv = gdf[gdf['NAME_UA_se'] == 'Львів']
#     lv = lv[lv.index == 14183]
#
#     pts = my_rand(PLANTS, lv['geometry'].values[0])
#     w = abs(np.random.normal(0, .01, POPULATION))
#     w = np.histogram(w, bins=PLANTS)[0]
#     pts['workers'] = w
#
#     hts = my_rand(HOUSES, lv['geometry'].values[0])
#     hts['family'] = 3
#
#     # ax = lv["geometry"].plot(alpha=.2,)
#     # hts.plot(ax=ax, color='black', alpha=.5, markersize=10)
#     # pts.plot(ax=ax, column='workers', alpha=1 , markersize= 50, legend=True, cmap = 'Greens')
#     # cx.add_basemap(ax, crs=lv.crs)
#     # plt.show()
#
#     lv["area"] = 0.1
#
#     m = lv.explore("area",
#                    scheme="naturalbreaks",  # use mapclassify's natural breaks scheme
#                    )
#     m = pts.explore("workers", m=m, cmap = 'Greys', # use red color on all points
#                     legend=True,
#      marker_kwds=dict(radius=10, fill=True), # make marker radius 10px with fill
#      tooltip_kwds=dict(labels=False), # do not show column label in the tooltip
#         )
#
#     m = hts.explore( m=m, color="red", # use red color on all points
#      marker_kwds=dict(radius=2, fill=True), # make marker radius 10px with fill
#      tooltip_kwds=dict(labels=False), # do not show column label in the tooltip
#                      )
#
#     folium.TileLayer('Stamen Toner', control=True).add_to(m)  # use folium to add alternative tiles
#     folium.LayerControl().add_to(m)  # use folium to add layer control
#     m.save('my_map.html')
# city_creation()
