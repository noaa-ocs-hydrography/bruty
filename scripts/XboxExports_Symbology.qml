<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.36.0-Maidenhead" styleCategories="Symbology">
  <renderer-v2 enableorderby="1" symbollevels="0" referencescale="-1" type="RuleRenderer" forceraster="0">
    <rules key="{1b1645df-c77c-4f39-80b6-ae0e58b962e7}">
      <rule filter="attribute(@feature, 'navigation') and attribute(@feature, 'internal') and attribute(@feature, 'public')" key="{5b2673e9-21fe-480a-9c9c-4e8be4d35ca1}" symbol="0" label="Complete"/>
      <rule filter="not(attribute(@feature, 'navigation') and attribute(@feature, 'internal') and attribute(@feature, 'public'))" key="{229e815b-d9f1-42b3-89b5-15800b1f8820}" symbol="1" label="Failed"/>
    </rules>
    <symbols>
      <symbol is_animated="0" clip_to_extent="1" alpha="0.5" force_rhr="0" type="fill" frame_rate="10" name="0">
        <data_defined_properties>
          <Option type="Map">
            <Option type="QString" value="" name="name"/>
            <Option name="properties"/>
            <Option type="QString" value="collection" name="type"/>
          </Option>
        </data_defined_properties>
        <layer id="{cdd7c3fa-6448-4006-9f80-5a3e2fdeb73c}" enabled="1" locked="0" pass="0" class="SimpleFill">
          <Option type="Map">
            <Option type="QString" value="3x:0,0,0,0,0,0" name="border_width_map_unit_scale"/>
            <Option type="QString" value="133,182,111,255,rgb:0.52156862745098043,0.71372549019607845,0.43529411764705883,1" name="color"/>
            <Option type="QString" value="bevel" name="joinstyle"/>
            <Option type="QString" value="0,0" name="offset"/>
            <Option type="QString" value="3x:0,0,0,0,0,0" name="offset_map_unit_scale"/>
            <Option type="QString" value="MM" name="offset_unit"/>
            <Option type="QString" value="35,35,35,255,rgb:0.13725490196078433,0.13725490196078433,0.13725490196078433,1" name="outline_color"/>
            <Option type="QString" value="solid" name="outline_style"/>
            <Option type="QString" value="0.26" name="outline_width"/>
            <Option type="QString" value="MM" name="outline_width_unit"/>
            <Option type="QString" value="solid" name="style"/>
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option type="QString" value="" name="name"/>
              <Option name="properties"/>
              <Option type="QString" value="collection" name="type"/>
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol is_animated="0" clip_to_extent="1" alpha="1" force_rhr="0" type="fill" frame_rate="10" name="1">
        <data_defined_properties>
          <Option type="Map">
            <Option type="QString" value="" name="name"/>
            <Option name="properties"/>
            <Option type="QString" value="collection" name="type"/>
          </Option>
        </data_defined_properties>
        <layer id="{36cc7d90-6d1c-4033-ae2d-7586ed1c4726}" enabled="1" locked="0" pass="0" class="SimpleFill">
          <Option type="Map">
            <Option type="QString" value="3x:0,0,0,0,0,0" name="border_width_map_unit_scale"/>
            <Option type="QString" value="200,0,3,255,hsv:0.99722222222222223,1,0.78431372549019607,1" name="color"/>
            <Option type="QString" value="bevel" name="joinstyle"/>
            <Option type="QString" value="0,0" name="offset"/>
            <Option type="QString" value="3x:0,0,0,0,0,0" name="offset_map_unit_scale"/>
            <Option type="QString" value="MM" name="offset_unit"/>
            <Option type="QString" value="35,35,35,255,rgb:0.13725490196078433,0.13725490196078433,0.13725490196078433,1" name="outline_color"/>
            <Option type="QString" value="solid" name="outline_style"/>
            <Option type="QString" value="0.26" name="outline_width"/>
            <Option type="QString" value="MM" name="outline_width_unit"/>
            <Option type="QString" value="solid" name="style"/>
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option type="QString" value="" name="name"/>
              <Option name="properties"/>
              <Option type="QString" value="collection" name="type"/>
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
    </symbols>
    <orderby>
      <orderByClause asc="1" nullsFirst="0">"tile_name"</orderByClause>
    </orderby>
  </renderer-v2>
  <selection mode="Default">
    <selectionColor invalid="1"/>
    <selectionSymbol>
      <symbol is_animated="0" clip_to_extent="1" alpha="1" force_rhr="0" type="fill" frame_rate="10" name="">
        <data_defined_properties>
          <Option type="Map">
            <Option type="QString" value="" name="name"/>
            <Option name="properties"/>
            <Option type="QString" value="collection" name="type"/>
          </Option>
        </data_defined_properties>
        <layer id="{93e6155f-8163-4f15-83d7-ab33792de781}" enabled="1" locked="0" pass="0" class="SimpleFill">
          <Option type="Map">
            <Option type="QString" value="3x:0,0,0,0,0,0" name="border_width_map_unit_scale"/>
            <Option type="QString" value="0,0,255,255,rgb:0,0,1,1" name="color"/>
            <Option type="QString" value="bevel" name="joinstyle"/>
            <Option type="QString" value="0,0" name="offset"/>
            <Option type="QString" value="3x:0,0,0,0,0,0" name="offset_map_unit_scale"/>
            <Option type="QString" value="MM" name="offset_unit"/>
            <Option type="QString" value="35,35,35,255,rgb:0.13725490196078433,0.13725490196078433,0.13725490196078433,1" name="outline_color"/>
            <Option type="QString" value="solid" name="outline_style"/>
            <Option type="QString" value="0.26" name="outline_width"/>
            <Option type="QString" value="MM" name="outline_width_unit"/>
            <Option type="QString" value="solid" name="style"/>
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option type="QString" value="" name="name"/>
              <Option name="properties"/>
              <Option type="QString" value="collection" name="type"/>
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
    </selectionSymbol>
  </selection>
  <blendMode>0</blendMode>
  <featureBlendMode>0</featureBlendMode>
  <layerGeometryType>2</layerGeometryType>
</qgis>
