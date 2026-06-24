CC = g++-13

FLAGS      = $(CVERSION) $(CFLAGS) $(OPTIMFLAGS) 
CFLAGS     = -Wall -Wextra
CVERSION   = -std=c++23
OPTIMFLAGS = -O3

OBJECTS = obj
INCLUDE = -I include

NETWORK_TESTS = network tensor matrix
N = 10

.PHONY: all clean run obj_dir

run:
	@echo "\033[34mCREATING EXECUTABLES\033[33m"
	@$(MAKE) --no-print-directory all
	@echo "\033[34mEXECUTING TESTS"
	@n=$(N); \
	w=$${#n}; \
	counter=0; \
	for i in $$(seq 1 $$n); do \
		printf "\033[0m[\033[33m%0$${w}d/%d\033[0m]: | " $$i $$n; \
		for f in $(OBJECTS)/*.test; do \
			out=$$( { ./$$f ; } 2>&1); \
			status=$$?; \
			if [ $$status -eq 0 ]; then \
				printf "\033[0m%s \033[32mPASSED\033[0m | " $$f; \
			else \
				printf "\033[0m%s \033[31mFAILED\033[0m\n |" "$$f"; \
				printf "\033[0mTrace: \033[31m"; \
				printf "%s" $$out; \
				counter=$$((counter+1)); \
				break; \
			fi; \
		done; \
		printf "\n"; \
	done; \
	if [ $$counter -eq 0 ]; then \
		printf "\033[34mALL TESTS \033[32mPASSED\033[0m\n"; \
	else \
		printf "\033[34mTESTS \033[31mFAILED\033[0m: \033[34m%d\033[0m\n" $$counter; \
	fi;

all: obj_dir $(addprefix $(OBJECTS)/, $(addsuffix .test, $(NETWORK_TESTS)))
	
obj_dir:
	@mkdir -p obj

$(OBJECTS)/%.test: src/test/%_test.cc
	$(CC) -MMD -MP $< -o $@ $(FLAGS) $(INCLUDE)

clean:
	@rm -f $(OBJECTS)/*

-include $(OBJECTS)/*.d