/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import org.apache.commons.lang.StringUtils;
import org.opensearch.Version;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.query.request.MethodParameter;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.File;
import java.util.Collections;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.BYTES_PER_KILOBYTES;
import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.EXPAND_NESTED;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.query.parser.RescoreParser.RESCORE_PARAMETER;

public class IndexUtil {

    public static final String MODEL_NODE_ASSIGNMENT_KEY = KNNConstants.MODEL_NODE_ASSIGNMENT;
    public static final String MODEL_METHOD_COMPONENT_CONTEXT_KEY = KNNConstants.MODEL_METHOD_COMPONENT_CONTEXT;

    private static final Version MINIMAL_SUPPORTED_VERSION_FOR_IGNORE_UNMAPPED = Version.V_2_11_0;
    private static final Version MINIMAL_SUPPORTED_VERSION_FOR_MODEL_NODE_ASSIGNMENT = Version.V_2_12_0;
    private static final Version MINIMAL_SUPPORTED_VERSION_FOR_MODEL_METHOD_COMPONENT_CONTEXT = Version.V_2_13_0;
    private static final Version MINIMAL_SUPPORTED_VERSION_FOR_RADIAL_SEARCH = Version.V_2_14_0;
    private static final Version MINIMAL_SUPPORTED_VERSION_FOR_METHOD_PARAMETERS = Version.V_2_16_0;
    private static final Version MINIMAL_SUPPORTED_VERSION_FOR_MODEL_VECTOR_DATA_TYPE = Version.V_2_16_0;
    private static final Version MINIMAL_RESCORE_FEATURE = Version.V_2_17_0;
    private static final Version MINIMAL_MODE_AND_COMPRESSION_FEATURE = Version.V_2_17_0;
    private static final Version MINIMAL_TOP_LEVEL_SPACE_TYPE_FEATURE = Version.V_2_17_0;
    private static final Version MINIMAL_SUPPORTED_VERSION_FOR_MODEL_VERSION = Version.V_2_17_0;
    private static final Version MINIMAL_EXPAND_NESTED_FEATURE = Version.V_2_19_0;
    // public so neural search can access it
    public static final Map<String, Version> minimalRequiredVersionMap = initializeMinimalRequiredVersionMap();
    public static final Set<VectorDataType> VECTOR_DATA_TYPES_NOT_SUPPORTING_ENCODERS = Set.of(VectorDataType.BINARY, VectorDataType.BYTE);

    /**
     * Determines the size of a file on disk in kilobytes
     *
     * @param filePath path to the file
     * @return file size in kilobytes
     */
    public static int getFileSizeInKB(String filePath) {
        if (filePath == null || filePath.isEmpty()) {
            return 0;
        }
        File file = new File(filePath);
        if (!file.exists() || !file.isFile()) {
            return 0;
        }

        return Math.toIntExact((file.length() / BYTES_PER_KILOBYTES) + 1L); // Add one so that integer division rounds up
    }

    /**
     * Validate that a field is a k-NN vector field and has the expected dimension
     *
     * @param indexMetadata metadata for index to validate
     * @param field field name to validate
     * @param expectedDimension expected dimension of the field. If this value is negative, dimension will not be
     *                          checked
     * @return ValidationException exception produced by field validation
     */
    @SuppressWarnings("unchecked")
    public static ValidationException validateKnnField(
        IndexMetadata indexMetadata,
        String field,
        int expectedDimension,
        VectorDataType trainRequestVectorDataType,
        KNNMethodContext trainRequestKnnMethodContext
    ) {
        // Index metadata should not be null
        if (indexMetadata == null) {
            throw new IllegalArgumentException("IndexMetadata should not be null");
        }

        ValidationException exception = new ValidationException();

        // Check the mapping
        MappingMetadata mappingMetadata = indexMetadata.mapping();
        if (mappingMetadata == null) {
            exception.addValidationError("Invalid index. Index does not contain a mapping");
            return exception;
        }

        // The mapping output *should* look like this:
        // "{properties={field={type=knn_vector, dimension=8}}}"
        Map<String, Object> properties = (Map<String, Object>) mappingMetadata.getSourceAsMap().get("properties");

        if (properties == null) {
            exception.addValidationError("Properties in map does not exists. This is unexpected");
            return exception;
        }

        // Check field path is valid
        if (StringUtils.isEmpty(field)) {
            exception.addValidationError(String.format(Locale.ROOT, "Field path is empty."));
            return exception;
        }

        Object fieldMapping = getFieldMapping(properties, field);

        // Check field existence
        if (fieldMapping == null) {
            exception.addValidationError(String.format("Field \"%s\" does not exist.", field));
            return exception;
        }

        // Check if field is a map. If not, that is a problem
        if (!(fieldMapping instanceof Map)) {
            exception.addValidationError(String.format("Field info for \"%s\" is not a map.", field));
            return exception;
        }

        Map<String, Object> fieldMap = (Map<String, Object>) fieldMapping;

        // Check fields type is knn_vector
        Object type = fieldMap.get("type");

        if (!(type instanceof String) || !KNNVectorFieldMapper.CONTENT_TYPE.equals(type)) {
            exception.addValidationError(String.format("Field \"%s\" is not of type %s.", field, KNNVectorFieldMapper.CONTENT_TYPE));
            return exception;
        }

        if (trainRequestVectorDataType != null) {
            VectorDataType trainIndexDataType = getVectorDataTypeFromFieldMapping(fieldMap);

            if (trainIndexDataType != trainRequestVectorDataType) {
                exception.addValidationError(
                    String.format(
                        Locale.ROOT,
                        "Field \"%s\" has data type %s, which is different from data type used in the training request: %s",
                        field,
                        trainIndexDataType.getValue(),
                        trainRequestVectorDataType.getValue()
                    )
                );
                return exception;
            }

            // Block binary and byte vector data type for any encoder
            if (trainRequestKnnMethodContext != null) {
                MethodComponentContext methodComponentContext = trainRequestKnnMethodContext.getMethodComponentContext();
                Map<String, Object> parameters = methodComponentContext.getParameters();

                if (parameters != null && parameters.containsKey(KNNConstants.METHOD_ENCODER_PARAMETER)) {
                    MethodComponentContext encoder = (MethodComponentContext) parameters.get(KNNConstants.METHOD_ENCODER_PARAMETER);
                    if (encoder != null
                        && VECTOR_DATA_TYPES_NOT_SUPPORTING_ENCODERS.contains(trainRequestVectorDataType)
                        && ENCODER_FLAT.equals(encoder.getName()) == false) {
                        exception.addValidationError(
                            String.format(
                                Locale.ROOT,
                                "encoder is not supported for vector data type [%s]",
                                trainRequestVectorDataType.getValue()
                            )
                        );
                        return exception;
                    }
                }
            }
        }

        // Return if dimension does not need to be checked
        if (expectedDimension < 0) {
            return null;
        }

        // Check that the dimension of the method passed in matches that of the model
        Object dimension = fieldMap.get(KNNConstants.DIMENSION);

        // If dimension is null, the training index/field could use a model. In this case, we need to get the model id
        // for the index and then fetch its dimension from the models metadata
        if (dimension == null) {
            throw new IllegalArgumentException("Dimension should not be null");
        }

        // If the dimension was found in training fields mapping, check that it equals the models proposed dimension.
        if ((Integer) dimension != expectedDimension) {
            exception.addValidationError(
                String.format(
                    "Field \"%s\" has dimension %d, which is different from " + "dimension specified in the training request: %d",
                    field,
                    dimension,
                    expectedDimension
                )
            );
            return exception;
        }

        return null;
    }

    /**
     * Gets the load time parameters for a given engine.
     *
     * @param spaceType Space for this particular segment
     * @param knnEngine Engine used for the native library indices being loaded in
     * @param indexName Name of OpenSearch index that the segment files belong to
     * @param vectorDataType Vector data type for this particular segment
     * @return load parameters that will be passed to the JNI.
     */
    public static Map<String, Object> getParametersAtLoading(
        SpaceType spaceType,
        KNNEngine knnEngine,
        String indexName,
        VectorDataType vectorDataType
    ) {
        Map<String, Object> loadParameters = Maps.newHashMap(ImmutableMap.of(SPACE_TYPE, spaceType.getValue()));

        loadParameters.put(VECTOR_DATA_TYPE_FIELD, vectorDataType.getValue());

        return Collections.unmodifiableMap(loadParameters);
    }

    public static boolean isClusterOnOrAfterMinRequiredVersion(String key) {
        Version minimalRequiredVersion = minimalRequiredVersionMap.get(key);
        if (minimalRequiredVersion == null) {
            return false;
        }
        return KNNClusterUtil.instance().getClusterMinVersion().onOrAfter(minimalRequiredVersion);
    }

    public static boolean isVersionOnOrAfterMinRequiredVersion(Version version, String key) {
        Version minimalRequiredVersion = minimalRequiredVersionMap.get(key);
        if (minimalRequiredVersion == null) {
            return false;
        }
        return version.onOrAfter(minimalRequiredVersion);
    }

    /**
     * Tell if it is binary index or not
     *
     * @param knnEngine knn engine associated with an index
     * @param parameters parameters associated with an index
     * @return true if it is binary index
     */
    public static boolean isBinaryIndex(KNNEngine knnEngine, Map<String, Object> parameters) {
        return parameters.get(VECTOR_DATA_TYPE_FIELD) != null
            && parameters.get(VECTOR_DATA_TYPE_FIELD).toString().equals(VectorDataType.BINARY.getValue());
    }

    /**
     * Update vector data type into parameters
     *
     * @param parameters parameters associated with an index
     * @param vectorDataType vector data type
     */
    public static void updateVectorDataTypeToParameters(Map<String, Object> parameters, VectorDataType vectorDataType) {
        if (VectorDataType.BINARY == vectorDataType) {
            parameters.put(VECTOR_DATA_TYPE_FIELD, vectorDataType.getValue());
        }
        if (VectorDataType.BYTE == vectorDataType) {
            parameters.put(VECTOR_DATA_TYPE_FIELD, vectorDataType.getValue());
        }
    }

    /**
     * This method retrieves the field mapping by a given field path from the index metadata.
     *
     * @param properties Index metadata mapping properties.
     * @param fieldPath The field path string that make up the path to the field mapping. e.g. "a.b.field" or "field".
     *                  The field path is applied and checked in OpenSearch, so it is guaranteed to be valid.
     *
     * @return           The field mapping object if found, or null if the field is not found in the index metadata.
     */
    private static Object getFieldMapping(final Map<String, Object> properties, final String fieldPath) {
        String[] fieldPaths = fieldPath.split("\\.");
        Object currentFieldMapping = properties;

        // Iterate through the field path list to retrieve the field mapping.
        for (String path : fieldPaths) {
            currentFieldMapping = ((Map<String, Object>) currentFieldMapping).get(path);
            if (currentFieldMapping == null) {
                return null;
            }

            if (currentFieldMapping instanceof Map<?, ?>) {
                Object possibleProperties = ((Map<String, Object>) currentFieldMapping).get("properties");
                if (possibleProperties instanceof Map<?, ?>) {
                    currentFieldMapping = possibleProperties;
                }
            }
        }

        return currentFieldMapping;
    }

    /**
     *  This method is used to get the vector data type from field mapping
     * @param fieldMap field mapping
     * @return vector data type
     */
    private static VectorDataType getVectorDataTypeFromFieldMapping(Map<String, Object> fieldMap) {
        if (fieldMap.containsKey(VECTOR_DATA_TYPE_FIELD)) {
            return VectorDataType.get((String) fieldMap.get(VECTOR_DATA_TYPE_FIELD));
        }
        return VectorDataType.DEFAULT;
    }

    /**
     * Initialize the minimal required version map
     *
     * @return minimal required version map
     */
    private static Map<String, Version> initializeMinimalRequiredVersionMap() {
        final Map<String, Version> versionMap = new HashMap<>() {
            {
                put("ignore_unmapped", MINIMAL_SUPPORTED_VERSION_FOR_IGNORE_UNMAPPED);
                put(MODEL_NODE_ASSIGNMENT_KEY, MINIMAL_SUPPORTED_VERSION_FOR_MODEL_NODE_ASSIGNMENT);
                put(MODEL_METHOD_COMPONENT_CONTEXT_KEY, MINIMAL_SUPPORTED_VERSION_FOR_MODEL_METHOD_COMPONENT_CONTEXT);
                put(KNNConstants.RADIAL_SEARCH_KEY, MINIMAL_SUPPORTED_VERSION_FOR_RADIAL_SEARCH);
                put(KNNConstants.METHOD_PARAMETER, MINIMAL_SUPPORTED_VERSION_FOR_METHOD_PARAMETERS);
                put(KNNConstants.MODEL_VECTOR_DATA_TYPE_KEY, MINIMAL_SUPPORTED_VERSION_FOR_MODEL_VECTOR_DATA_TYPE);
                put(RESCORE_PARAMETER, MINIMAL_RESCORE_FEATURE);
                put(KNNConstants.MINIMAL_MODE_AND_COMPRESSION_FEATURE, MINIMAL_MODE_AND_COMPRESSION_FEATURE);
                put(KNNConstants.TOP_LEVEL_SPACE_TYPE_FEATURE, MINIMAL_TOP_LEVEL_SPACE_TYPE_FEATURE);
                put(KNNConstants.MODEL_VERSION, MINIMAL_SUPPORTED_VERSION_FOR_MODEL_VERSION);
                put(EXPAND_NESTED, MINIMAL_EXPAND_NESTED_FEATURE);
            }
        };

        for (final MethodParameter methodParameter : MethodParameter.values()) {
            if (methodParameter.getVersion() != null) {
                versionMap.put(methodParameter.getName(), methodParameter.getVersion());
            }
        }
        return Collections.unmodifiableMap(versionMap);
    }

    /**
     * Tell if it is byte index or not
     *
     * @param parameters parameters associated with an index
     * @return true if it is binary index
     */
    public static boolean isByteIndex(Map<String, Object> parameters) {
        return parameters.getOrDefault(VECTOR_DATA_TYPE_FIELD, VectorDataType.DEFAULT.getValue())
            .toString()
            .equals(VectorDataType.BYTE.getValue());
    }
}
