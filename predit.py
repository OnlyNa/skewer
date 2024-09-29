import json

from ultralytics import YOLO
import cv2


# 定义函数提取 YOLOv8 结果并转换为 JSON 格式
def result_to_json(result, class_names: dict):
	# 提取关键信息
	output = []
	for box, score, label in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
		output.append({
			"box": box.tolist(),  # 边界框坐标
			"score": float(score),  # 置信度
			"label": class_names[int(label)]  # 类别标签
		})

	# 转换为 JSON 格式
	return json.dumps(output, indent=4)


model = YOLO("best.pt")

image = cv2.imread("xrXbwTqpr3-xLZGcO-IYHw.jpg")
results = model.predict(source=[image], save=False, show=False, conf=0.5)  # save plotted images
json_result = result_to_json(results[0], dict(model.names))

# # 查看结果并绘制边界框
# for r in results:
# 	boxes = r.boxes  # 获取 Boxes 对象
#
# 	for idx, box in enumerate(boxes.xyxy):  # 遍历每个边界框
# 		# 输出此框的置信度
# 		if idx + 1 == len(boxes.xyxy):
# 			color = (0, 0, 255)
# 		else:
# 			color = (0, 255, 0)
#
# 		x1, y1, x2, y2 = map(int, box)  # 转换为整数坐标
#
# 		# 在图像上绘制边界框
# 		cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)
# 		# 计算中心点坐标
# 		center_x = int((x1 + x2) / 2)
# 		center_y = int((y1 + y2) / 2)
#
# 		# # 在框中间写入序号
# 		# cv2.putText(image, str(idx + 1), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 2,
# 		#             lineType=cv2.LINE_AA)

# 保存或显示结果图像
# 直接在显示时调整图像大小
# display_width = 1000
# resized_image = cv2.resize(image, (display_width, int(image.shape[0] * (display_width / image.shape[1]))))
# cv2.imshow("Detected Objects", resized_image)  # 显示结果图像
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# from ndarray
# im2 = cv2.imread("ls.jpg")
# results = model.predict(source=im2,save=True,save_txt=True) # save predictions as labels
# #from list of PIL/ndancay
# results = model. predict(source=[im1, im2])
