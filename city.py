import numpy as np
import geopandas
from shapely.geometry import Point
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class city:
    def __init__(self, geom, population, plants, houses, proba_infection, sickness_day, proba_death, isolation,
                 without_symptom, transport, vaccination, security, second_wave, contagiousness,
                 restaurant, file = False):
        self.geom = geom
        self.population = population
        self.plants = plants
        self.houses = houses
        self.transport = self.transport_coef = transport
        self.restaurant = self.restaurant_coef = restaurant
        self.proba_infection = proba_infection
        self.sickness_day = sickness_day
        self.proba_death = proba_death
        self.isolation = isolation
        self.without_symptom = without_symptom
        self.vaccination = vaccination
        self.map = self.init_city(file)
        self.security = security
        self.second_wave = second_wave
        self.contagiousness = contagiousness

    def prob_contagiousness(self, day):
        c, a, b = self.contagiousness
        if day <= c:
            x = (c - day) / a
            res = max(0, ((1 - x ** 2) ** 0.5).real)
        else:
            x = (day - c) / b
            res = np.exp(-abs(x) ** 3)
        return res

    def infected_security(self, x):
        sec = 1 / (1 + np.exp(- self.security[1] * (x -  self.security[2])))  #  sec = 1 / (1 + np.exp(-50 * (x - .1))) + 40 days - plato
        return sec

    def init_transport(self, sec = 0):
        # Transport creation
        tr = int(len(self.population) * self.transport_coef * (1-sec))
        self.transport = pd.DataFrame([tr] + [1] * (len(self.population) - tr), columns=["people"])
        self.transport["Type"] = "Car"
        self.transport.loc[0, 'Type'] = "Bus"
        p = [[i] * p for i, p in self.transport.people.items()]
        p_tr = []
        for h in p:
            p_tr += h
        np.random.shuffle(p_tr)
        self.population["Transport"] = p_tr
        self.population["peoples_tr"] = \
        self.population.merge(self.transport[['people']], left_on='Transport', right_index=True)['people']

    def init_restaurant(self, sec = 0):
        # restaurant creation
        tr = int(len(self.population) * self.restaurant_coef * (1-sec))
        self.restaurant = pd.DataFrame([tr] + [1] * (len(self.population) - tr), columns=["people"])
        self.restaurant["Type"] = "Alone"
        self.restaurant.loc[0, 'Type'] = "Restaurant"
        p = [[i] * p for i, p in self.restaurant.people.items()]
        p_tr = []
        for h in p:
            p_tr += h
        np.random.shuffle(p_tr)
        self.population["Restaurant"] = p_tr
        self.population["peoples_rest"] = \
        self.population.merge(self.restaurant[['people']], left_on='Restaurant', right_index=True)['people']

    def init_city(self, file):
        if file == False:

            def my_rand(number, geom):
                '''
                Ініціалізація випадкових координат в межаї міста
                :param number: число обʼєктів
                :param geom: геометрія міста
                :return:
                '''
                xmin, ymin, xmax, ymax = geom.bounds
                pts = []
                for p in range(number):
                    while True:
                        x = (xmax - xmin) * np.random.random() + xmin
                        y = (ymax - ymin) * np.random.random() + ymin
                        if Point(x, y).intersects(geom):
                            pts.append(Point(x, y))
                            break
                pts = geopandas.GeoDataFrame(pts, columns=['geometry']).set_crs('EPSG:4326').to_crs('EPSG:3857')
                return pts

            # Plants generation
            pts = my_rand(self.plants, self.geom)
            w = abs(np.random.normal(0, .01, self.population))
            #Fig 1 pop 10000 plants 50
            # ax= sns.histplot(w, bins=self.plants)
            # ax.set(xlabel='Probability')
            # plt.show()
            # pd.DataFrame(np.histogram(w, bins=self.plants)).T.to_clipboard()
            # exit()

            w = np.histogram(w, bins=self.plants)[0]
            pts['workers'] = w
            self.plants = pts

            # Houses creation
            hts = my_rand(self.houses, self.geom)
            w = np.random.random(self.population)
            #Fig 2 pop 100 plants 10
            # ax= sns.histplot(w, bins=self.houses)
            # ax.set(xlabel='Probability')
            # plt.show()
            # pd.DataFrame(np.histogram(w, bins=self.houses)).T.to_clipboard()
            # exit()
            w = np.histogram(w, bins=self.houses)[0]
            hts['family'] = w
            self.houses = hts

            # Population creation
            p = [[i]*p for i, p  in self.houses.family.items()]
            p_f =[]
            for h in p:
                p_f += h
            np.random.shuffle(p_f)
            population = pd.DataFrame({'House': p_f,
                                       'Plant': None,
                                       'Distance': None,
                                       'State': 'S',
                                       'sickness_day': 0,
                                       'proba_death': np.random.rand(self.population),
                                       'isolation': False,
                                       'symptom': [np.random.rand() > self.without_symptom for _ in range(self.population)]})

            for i, p in population.iterrows():
                h_l = self.houses.loc[p.House, 'geometry'] # House coordinates
                self.plants['distance'] = self.plants['geometry'].distance(h_l)
                for j, pl in self.plants.sort_values(by=['distance']).iterrows():
                    workers = len(population[population.Plant == j])
                    if workers < pl.workers:
                        population.loc[i, "Plant"] = j
                        population.loc[i, "Distance"] = pl.distance
                        break
            self.population = population
            self.plants.to_file('plants.geojson', driver='GeoJSON')
            self.population.to_csv('population.csv')
            self.houses.to_file('houses.geojson', driver='GeoJSON')
        else:
            self.plants = geopandas.read_file('plants.geojson')
            self.population = pd.read_csv('population.csv', index_col=0)
            self.houses = geopandas.read_file('houses.geojson')

        # sns.histplot(self.population.Distance)
        # plt.show()
        # exit()
        self.population["peoples_hs"] = self.population.merge(self.houses[['family']], left_on='House', right_index = True)['family']
        self.population["peoples_pl"] = self.population.merge(self.plants[['workers']], left_on='Plant', right_index = True)['workers']

        self.population["symptom"] = [np.random.rand() > self.without_symptom for _ in range(len(self.population))]

        # transport
        self.init_transport()
        # restaurant
        self.init_restaurant()
        #vaccination
        if self.vaccination[0]:
            self.population["immunitas"] = [self.vaccination[2] if np.random.rand() < self.vaccination[1] else 0 for _ in range(len(self.population))]
        else:
            self.population["immunitas"] = 0


        print(self.plants)
        print(self.houses)
        print(self.transport)
        print(self.restaurant)
        # sns.histplot(self.plants.workers, bins=len(self.plants.workers.value_counts()))
        # plt.show()
        # print(self.plants.workers.value_counts().sort_index())
        # pd.DataFrame(np.histogram(self.plants.workers, bins=len(self.plants.workers.value_counts()))).T.to_clipboard()

        # sns.histplot(self.houses.family, bins=len(self.houses.family.value_counts()))
        # plt.show()
        # print(self.houses.family.value_counts().sort_index())
        # pd.DataFrame(np.histogram(self.houses.family, bins=len(self.houses.family.value_counts()))).T.to_clipboard()


        # sns.histplot(self.population.Distance)
        # plt.show()
        # print(self.population.Distance.value_counts().sort_index())
        # pd.DataFrame(np.histogram(self.population.Distance, bins=100)).T.to_clipboard()
        # exit()
        print(self.population)


    def person_isolation(self):
        if sum(self.population.State == "I") >= self.isolation[3]:
            i_lim = (self.population.State == "I") & (self.population.isolation == False) & (self.population.sickness_day >= self.isolation[2]) & (self.population.symptom)
            lim_new_isolate = i_lim & (np.random.rand(len(self.population)) < self.isolation[1])
            self.population.loc[lim_new_isolate, 'isolation'] = True

    def infection(self, col, col_pl, pow):
        inf = self.population[(self.population.State == "I") & (self.population.isolation == False)] # infected person
        inf_pl = pd.unique(inf[col]) #infected objects
        id_sick_candidate = self.population[col].isin(inf_pl) & (self.population.State == "S") # sick candidate
        # print(id_sick_candidate, )
        # print(sum(id_sick_candidate))
        # print(inf_pl)
        # print(self.population.merge(self.plants[['workers']], left_on=col, right_index = True))

        # self.population["peoples"] = self.population.merge(pl[[col_pl]], left_on=col, right_index = True)[col_pl]
        if self.security[0]:
            x = len(self.population[self.population.State=="I"])/len(self.population)
            sec = self.infected_security(x)
        else:
            sec = 0
        # print(sec)
        self.population["il_people"] = self.population.merge(pd.pivot_table(self.population, index=[col], columns=['State'] , values= 'contagiousness', aggfunc=np.sum), left_on=col, right_index = True)["I"]
        # print(self.population["il_people"].max(), self.population["contagiousness"].max())
        # self.population["il_people"] = self.population.merge(pd.crosstab(self.population[col], self.population.State), left_on=col, right_index = True)["I"]
        new_I = id_sick_candidate & (np.random.rand(len(self.population)) < (pow*(1-sec)*self.proba_infection*(1-self.population["immunitas"])*self.population.il_people*10/self.population[col_pl]))
        self.population.loc[new_I, 'State'] = "I"
        self.population.loc[new_I, 'sickness_day'] = 0

        # self.population.loc[id_sick_candidate & (np.random.rand(len(self.population)) < self.proba_infection), 'State'] = "I"


    def move(self, day):

        print('Day=', day)
        # if day==30:
        #     st = pd.crosstab(self.population['Plant'], self.population.State)
        #     print(st[st.I>0])
        #     st2 = pd.pivot_table(self.population, index=['Plant'], columns=['State'] , values= 'contagiousness', aggfunc=np.sum)
        #     print(st2[st.I>0])
        #     # print("Test", self.population.sickness_day.apply(self.prob_contagiousness))
        #     exit()

        self.population.loc[:, 'contagiousness'] = 0
        iln = self.population.State=="I"
        self.population.loc[iln, 'contagiousness'] = self.population.loc[iln, "sickness_day"].apply(self.prob_contagiousness)
        # print(self.population.loc[iln, 'contagiousness'])
        # print(self.population[iln][['contagiousness','isolation']] )



        if self.isolation[0]:
            self.person_isolation()

        if self.security:
            x = len(self.population[self.population.State=="I"])/len(self.population)
            sec = self.infected_security(x)
            self.init_transport(sec)

        # weekend
        # if not (day % 7 == 6 or day % 7 == 0):
        #     print("Job")
        #     self.infection('Transport', 'peoples_tr', 1)
        #     self.infection('Plant', 'peoples_pl', 8)
        #     self.infection('Restaurant', 'peoples_rest', 1)
        #     self.infection('Transport', 'peoples_tr', 1)
        # else:
        #     print("Weekend")

        self.infection('Transport', 'peoples_tr', 1)
        self.infection('Plant', 'peoples_pl', 8)
        self.infection('Restaurant', 'peoples_rest', 1)
        self.infection('Transport', 'peoples_tr', 1)
        self.infection('House', 'peoples_hs', 12)
        # self.infection('Plant', self.plants, 'workers')
        # self.infection('House', self.houses, 'family')
        # #
        # self.population.loc[self.population.State == "I", 'sickness_day'] +=  1

        r_lim = (self.population.sickness_day >= self.sickness_day) & \
                (self.population.State == "I") & \
                (self.population.proba_death >= self.proba_death)
        self.population.loc[r_lim, 'State'] = "R"
        self.population.loc[r_lim, 'Isolation'] = False
        self.population.loc[r_lim, 'sickness_day'] = 0

        d_lim = (self.population.sickness_day >= self.sickness_day) & \
                (self.population.State == "I") & \
                (self.population.proba_death < self.proba_death)
        self.population.loc[d_lim, 'State'] = "D"
        self.population.loc[d_lim, 'Isolation'] = False
        self.population.loc[d_lim, 'sickness_day'] = 0

        # second wave
        if self.second_wave[0]:
            sw_lim = (self.population.sickness_day >= self.second_wave[1]) & \
                    (self.population.State == "R")
            self.population.loc[sw_lim, 'State'] = "S"
            self.population.loc[r_lim, 'Isolation'] = False
            self.population.loc[r_lim, 'sickness_day'] = 0

        self.population.loc[:, 'sickness_day'] +=  1





