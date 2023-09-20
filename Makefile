TORCHINFO_REPORT := torchinfo_report.txt


$(TORCHINFO_REPORT):
	python make_torchinfo_report.py > $(TORCHINFO_REPORT)


clean:
	rm $(TORCHINFO_REPORT)


all:
	$(TORCHINFO_REPORT)
