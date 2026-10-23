# Thanh phan ESP-IDF: thu vien trich xuat dac trung + OC-SVM phan mem.
# Truoc khi build: chep outputs/fpga/ocsvm_params.h va ocsvm_model_data.h vao thu muc nay
# (hoac dat bien OCSVM_GEN_DIR tro toi outputs/fpga).
if(NOT DEFINED OCSVM_GEN_DIR)
    set(OCSVM_GEN_DIR ${CMAKE_CURRENT_LIST_DIR})
endif()
idf_component_register(SRCS "ocsvm_features.c" "ocsvm_soft.c"
                       INCLUDE_DIRS "." ${OCSVM_GEN_DIR})
target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
