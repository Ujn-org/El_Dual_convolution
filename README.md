# Xian_congetation_pre_competation
## data description:
  https://outreach.didichuxing.com/app-vue/DatasetProjectDetail?id=1022
  link  小段的id	
  label	对应时间的link的路况状态	
  current_slice_id	当前时间片id	
  future_slice_id	待预测时间片id	
  recent_feature	近期n个时间片路况特征，n=5，时间片之间空格分隔，字段之间,分隔. 具体格式：时间片:路况速度,eta速度,路况状态,参与路况计算的车辆数. 特征都为0时，说明此时间片无车经过	
  history_feature	历史同期n个时间片路况特征，星期之间;分隔，共4组（-28,-21,-14,-7），每组格式和recent_feature一致	
