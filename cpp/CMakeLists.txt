set(GEN_SOURCES utils.cpp utils.h geometry.cpp geometry.h range.h mat2f.cpp mat2f.h)


convertIntoHeader(cl/avc.cl cl/avc_cl.h avc_kernel_sources)
convertIntoHeader(cl/voronoi.cl cl/voronoi_cl.h kernel_sources)
add_executable(VoronoiClassifier_cl
        main_vc.cpp AbstractVoronoiClassifierGPU.cpp AbstractVoronoiClassifierGPU.h cl/avc_cl.h cl/avc.cl
        VoronoiClassifier_cl.cpp VoronoiClassifier_cl.h cl/voronoi_cl.h cl/voronoi.cl
        weights.cpp weights.h
        ${GEN_SOURCES})
target_link_libraries(VoronoiClassifier_cl libclew libgpu libutils cnpy)
#add_custom_command(TARGET VoronoiClassifier_cl POST_BUILD
#        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:VoronoiClassifier_cl> ${PROJECT_SOURCE_DIR})


add_executable(VoronoiClassifier_kd
        main_vc.cpp VoronoiClassifier_kd.cpp VoronoiClassifier_kd.h
        KDTree.cpp KDTree.h
        weights.cpp weights.h
        ${GEN_SOURCES})
target_link_libraries(VoronoiClassifier_kd libutils cnpy)
target_compile_definitions(VoronoiClassifier_kd PUBLIC VCTYPE=KD)
#add_custom_command(TARGET VoronoiClassifier_kd POST_BUILD
#        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:VoronoiClassifier_kd> ${PROJECT_SOURCE_DIR})


add_executable(shortest_distances
        tools/_shortest_distances.cpp ${GEN_SOURCES})
target_link_libraries(shortest_distances cnpy)
set_target_properties(shortest_distances PROPERTIES EXCLUDE_FROM_ALL TRUE)