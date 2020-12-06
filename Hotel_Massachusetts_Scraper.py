#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 21:42:01 2020

@author: sophie
"""

#import libraries
import os
import pandas as pd
import time
from bs4 import BeautifulSoup
from selenium import webdriver
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 5)
pd.set_option('display.width',800)

#%%%% driver
os.chdir('/Users/sophie/Desktop/BUS 256 - Marketing Analytics/Final Project')
driver = webdriver.Chrome(executable_path="/Users/sophie/Desktop/Chromedriver_max/chromedriver")
scrape_link = 'https://www.tripadvisor.com/Hotels-g28942-Massachusetts-Hotels.html'
driver.get(scrape_link)

time.sleep(10)

#%%%%% calendar selection
#click check-in date
driver.find_element_by_xpath("//div[@aria-label = 'Sun May 02 2021']").click()

time.sleep(2)

#click check-out date
driver.find_element_by_xpath("//div[@aria-label = 'Wed May 05 2021']").click()

#%%%% 

hotels_list = []
time.sleep(3)

for i in range(16):
    #find all the reviews in one page
    time.sleep(3)
    hotels = driver.find_elements_by_xpath("//div[@class = 'ui_column is-8 main_col allowEllipsis']")
          
    for r in range(len(hotels)):
        one_hotel = {}
        time.sleep(1)
        soup = BeautifulSoup(hotels[r].get_attribute('innerHTML'))
        
        parentHandle = driver.current_window_handle

        #extract hotel names
        try:
            one_hotel_name = soup.find('div', attrs={'class':'listing_title'}).text
        except:
            one_hotel_name = ""
        one_hotel['hotel_name'] = one_hotel_name
        
        
        #extract hotel id
        try:
            one_hotel_id = soup.find('a')["id"]
        except:
            one_hotel_id = ""
        one_hotel['id'] = one_hotel_id   
        
        
        #extract hotel prices
        try:
            one_hotel_price = int(soup.find('div', attrs={'class':'price __resizeWatch'}).text[1:])
        except:
            one_hotel_price = 0
        one_hotel['hotel_price'] = one_hotel_price
        
        
        #extract hotel review numbers
        try:
            nreview = soup.findAll('a')[2].text
            one_hotel_nreview = int(nreview[:nreview.find(" ")].replace(",",""))
        except:
            one_hotel_nreview = 0
        one_hotel['hotel_nreview'] = one_hotel_nreview 
        
        
        #extract one review from each hotel
        try:
            one_hotel_review = soup.findAll('a')[3].text
        except:
            one_hotel_review = ""
        one_hotel['hotel_one_review'] = one_hotel_review
              
        #extract ratings
        try:
            rating = str(soup.findAll('a')[1])
            one_hotel_rating = float(rating[rating.find('"')+1:rating.find(" of")])
        except:
            one_hotel_rating = 0
        one_hotel['hotel_rating'] = one_hotel_rating                      
        
        time.sleep(1)
        one_hotel_link = driver.find_element_by_xpath("//*[@id='{}']".format(one_hotel_id))
        time.sleep(1)
    
        try:
            #click each hotel link and switch to its web page
            one_hotel_link.click()
            time.sleep(1)

            handles = driver.window_handles
            driver.switch_to.window(handles[1])
            time.sleep(1)

            #create a new dict for storing info from each webpage
            new = {}
            
            #extract great_for_walker_rating
            try:
                great_walker = float(driver.find_element_by_xpath("//span[@class='oPMurIUj _1iwDIdby']").get_attribute("innerHTML"))                   
            except:
                great_walker = 0
            new['great_for_walker_rating'] = great_walker 
            

            #extract how many restaurants_nearby
            try:
                restaurants_nearby = int(driver.find_element_by_xpath("//span[@class='oPMurIUj TrfXbt7b']").get_attribute("innerHTML"))                
            except:
                restaurants_nearby = 0
            new['restaurants_nearby'] = restaurants_nearby

            
            try:
                attractions = int(driver.find_element_by_xpath("//span[@class='oPMurIUj _1WE0iyL_']").get_attribute("innerHTML"))               
            except:
                attractions = 0
            new['nearby_attractions'] = attractions

            try:    
                ranking = driver.find_element_by_xpath("//span[@class = '_28eYYeHH']").get_attribute("innerHTML")             
            except:
                ranking = ""
            new['ranking'] = ranking  
            
            try:
                one_hotel_description = driver.find_elements_by_xpath("//div[@class = 'cPQsENeY']")[0].get_attribute("innerHTML")
            except:
                one_hotel_description = ""
            new['description'] = one_hotel_description
            
            try:    
                location_rating = driver.find_elements_by_xpath("//div[@class = '_1krg1t5y']")[0].get_attribute("innerHTML")
                rating_index = location_rating.find(" bubble_")
                location_rating = float(location_rating[rating_index+8:rating_index+10])*0.1
            except:
                location_rating = 0
            new['location_rating'] = location_rating

            try:    
                cleanliness_rating = driver.find_elements_by_xpath("//div[@class = '_1krg1t5y']")[1].get_attribute("innerHTML")
                rating_index = cleanliness_rating.find(" bubble_")
                cleanliness_rating = float(cleanliness_rating[rating_index+8:rating_index+10])*0.1
            except:
                cleanliness_rating = 0
            new['cleanliness_rating'] = cleanliness_rating

            try:    
                service_rating = driver.find_elements_by_xpath("//div[@class = '_1krg1t5y']")[2].get_attribute("innerHTML")             
                rating_index = service_rating.find(" bubble_")
                service_rating = float(service_rating[rating_index+8:rating_index+10])*0.1

            except:
                service_rating_html = 0
            new['service_rating'] = service_rating

            try:    
                value_rating = driver.find_element_by_xpath("//div[@class = '_1krg1t5y QrxaKkaW']").get_attribute("innerHTML") 
                rating_index = value_rating.find(" bubble_")
                value_rating = float(value_rating[rating_index+8:rating_index+10])*0.1

            except:
                value_rating = 0
            new['value_rating'] = value_rating
            
            #try:    
                #hotel_class = driver.find_element_by_xpath("//span[@class = 'cwu1UFvH']").get_attribute("innerHTML")
                #index = hotel_class.find(" of 5 bubbles")
                #hotel_class = int(hotel_class[index - 3:index])
            #except:
                #hotel_class = 0
            #new['hotel_class'] = hotel_class
            
            try:    
                QA_number = driver.find_elements_by_xpath("//span[@class = '_1aRY8Wbl']")[1].get_attribute("innerHTML")
                QA_number = int(QA_number.replace(",",""))
            except:
                QA_number = 0
            new['QA_number'] = QA_number

            try:    
                room_tips_number = driver.find_elements_by_xpath("//span[@class = '_1aRY8Wbl']")[2].get_attribute("innerHTML")
            except:
                room_tips_number = 0
            new['room_tips_number'] = room_tips_number

            driver.close()             
            driver.switch_to.window(parentHandle)
            
        except:
            new = {}
             
        one_hotel.update(new)
        hotels_list.append(one_hotel)

    #click the arrow if it is not None, and give 2s time for loading    
    try:
        arrow = driver.find_element_by_xpath("//*[@id='taplc_main_pagination_bar_dusty_hotels_resp_0']/div/div/div/span[2]")
        arrow.click()       
        time.sleep(2)      

    except:
        break

#driver.close()  
dta = pd.DataFrame.from_dict(hotels_list)
dta.to_csv("Massachusetts_hotel.csv")

#%%%% 
