#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 21:44:40 2020

@author: sophie
"""

import pandas as pd
import os
os.chdir('/Users/sophie/Desktop/BUS 256 - Marketing Analytics/Final Project/Massachusetts/')
dta = pd.read_csv("Massachusetts_hotel.csv", index_col = False)

#replace 0 with the mean of the column
dta["hotel_name"] = dta["hotel_name"].str.replace("Sponsored","").str.strip()

price_mean = round(dta[dta["hotel_price"]!=0]["hotel_price"].mean())
dta["hotel_price"] = dta["hotel_price"].replace(0,price_mean)

nreview_mean = round(dta[dta["hotel_nreview"]!=0]["hotel_nreview"].mean())
dta["hotel_nreview"] = dta["hotel_nreview"].replace(0,nreview_mean)

rating_mean = round(dta[dta["hotel_rating"]!=0]["hotel_rating"].mean())
dta["hotel_rating"] = dta["hotel_rating"].replace(0,rating_mean)

dta["id"]  = dta["id"].str.replace("property_","").astype("int64")

gfw_mean = round(dta[dta["great_for_walker_rating"]!=0]["great_for_walker_rating"].mean())
dta["great_for_walker_rating"] = dta["great_for_walker_rating"].replace(0,gfw_mean)

rest_mean = round(dta[dta["restaurants_nearby"]!=0]["restaurants_nearby"].mean())
dta["restaurants_nearby"] = dta["restaurants_nearby"].replace(0,rest_mean)

nbattr_mean = round(dta[dta["nearby_attractions"]!=0]["nearby_attractions"].mean())
dta["nearby_attractions"] = dta["nearby_attractions"].replace(0,nbattr_mean)

locrat_mean = round(dta[dta["location_rating"]!=0]["location_rating"].mean())
dta["location_rating"] = dta["location_rating"].replace(0,locrat_mean)

clean_mean = round(dta[dta["cleanliness_rating"]!=0]["cleanliness_rating"].mean())
dta["cleanliness_rating"] = dta["cleanliness_rating"].replace(0,clean_mean)

service_mean = round(dta[dta["service_rating"]!=0]["service_rating"].mean())
dta["service_rating"] = dta["service_rating"].replace(0,service_mean)

value_mean = round(dta[dta["value_rating"]!=0]["value_rating"].mean())
dta["value_rating"] = dta["value_rating"].replace(0,value_mean)

qa_mean = round(dta[dta["QA_number"]!=0]["QA_number"].mean())
dta["QA_number"] = dta["QA_number"].replace(0,qa_mean)

tip_mean = round(dta[dta["room_tips_number"]!=0]["room_tips_number"].mean())
dta["room_tips_number"] = dta["room_tips_number"].replace(0,tip_mean)

dta.to_csv("Massachusetts_hotels.csv")

