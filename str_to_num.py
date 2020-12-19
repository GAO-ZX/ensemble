import pandas as pd
hotel=pd.read_csv('hotel_bookings.csv')
hotel['agent'].fillna('0',inplace=True)
hotel['company'].fillna('0',inplace=True)
hotel=hotel.drop(['assigned_room_type', 'reservation_status', 'reservation_status_date'],axis=1)
hotel.dropna(inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in hotel.columns.values.tolist():
 if i=='is_canceled' or i=='lead_time' or i=='arrival_date_year' or i=='arrival_date_week_number' or i=='arrival_date_day_of_month' or i=='stays_in_weekend_nights' or i=='stays_in_week_nights' or i=='adults' or i=='children' or i=='babies' or i=='is_repeated_guest' or i=='previous_cancellations' or i=='previous_bookings_not_canceled' or i=='booking_changes' or i=='agent' or i=='company' or i=='days_in_waiting_list' or i=='adr' or i=='required_car_parking_spaces' or i=='total_of_special_requests':
  pass
 else:
  hotel[i] = le.fit_transform(hotel[i])

hotel_target = hotel['is_canceled']
hotel_data=hotel.drop(['is_canceled'],axis=1)
yX=hotel_target
yX=pd.concat([yX,hotel_data],axis=1)
yX.to_csv('temp.csv',encoding='utf-8',index=False)