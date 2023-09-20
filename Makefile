MODEL_SUMMARY := model_summary.txt
COMPUTE_GRAPH := compute_graph.pdf


all: $(MODEL_SUMMARY) $(COMPUTE_GRAPH)

$(MODEL_SUMMARY):
	python make_model_summary.py > $(MODEL_SUMMARY)


$(COMPUTE_GRAPH):
	python make_compute_graph.py
	rm Digraph.gv
	mv Digraph.gv.pdf $(COMPUTE_GRAPH)


clean:
	rm $(MODEL_SUMMARY) $(COMPUTE_GRAPH)


