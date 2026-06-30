from dataretrieval import waterdata
import os

### Set site list
# site_list = ['USGS-460135118401101','USGS-460109118262801','USGS-460043118232801','USGS-460014118281805','USGS-460014118282201','USGS-460235118330701'] 
#   use timespan = '2022-07-01/2024-09-01'
site_list = ['USGS-460014118281805'] 
#   use timespan = '2023-01-01/2024-09-01'

# site_list = ['USGS-461935118081501'] # Whetstone, CRN well with daily values

# example
# site_list = ['USGS-464017120282501', 'USGS-464042120272801'] 

### API key
os.environ["API_USGS_PAT"] = "46fCmfu2RJt5c4m027qU1LOOWPvBp5mldEm3APhZ"

### Parameter codes
# pcode = '62611' # Groundwater level above NAVD 1988, feet
pcode = '72019' # Depth to water level, feet below land surface

timespan = '2023-01-01/2024-09-01'

### get basic site info
# df_nwis = waterdata.get_monitoring_locations(monitoring_location_id = site_list)[0]
# df_nwis.to_csv('site_info.csv', index=False)

### get discrete WL
# df_discrete_WL = waterdata.get_field_measurements(monitoring_location_id = site_list, parameter_code = pcode,
#                  time = timespan)[0]
# df_discrete_WL.to_csv('discrete_WL.csv', index=False)				 
				 
### get daily WL
# df_daily_WL = waterdata.get_daily(
#                 monitoring_location_id=site_list,
#                 parameter_code=pcode,
#                 time = timespan
#             )[0]
# df_daily_WL.to_csv('daily_WL.csv', index=False)

### get continuous WL
df_cont_WL = waterdata.get_continuous(
                monitoring_location_id=site_list,
                parameter_code=pcode,
                time = timespan
            )[0]            
df_cont_WL.to_csv('continuous_WL.csv', index=False)




