<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="Actions" version="3.28.3-Firenze">
  <attributeactions>
    <defaultAction value="{d5ea2536-1039-4645-a609-6b45c0bb948a}" key="Canvas"/>
    <actionsetting notificationMessage="" capture="0" shortTitle="Combine" icon="" type="1" isEnabledOnlyWhenEditable="0" id="{d5ea2536-1039-4645-a609-6b45c0bb948a}" name="Request Combine" action="from qgis.utils import iface&#xd;&#xa;from qgis.core import QgsMessageLog&#xd;&#xa;layer = QgsProject().instance().mapLayer(&quot;[%@layer_id%]&quot;)&#xd;&#xa;field = layer.fields().lookupField(&quot;request_combine&quot;)&#xd;&#xa;if bool(&quot;[%$id%]&quot;):&#xd;&#xa;    selected_ids = [[%$id%]]&#xd;&#xa;else:&#xd;&#xa;    selected_ids = [feature.id() for feature in layer.selectedFeatures()]    &#xd;&#xa;QgsMessageLog.logMessage(&quot;combine:&quot;+str(selected_ids))&#xd;&#xa;with edit(layer):&#xd;&#xa;    for sid in selected_ids:&#xd;&#xa;        # current = layer.getFeature([%$id%]).attribute(field)&#xd;&#xa;        layer.changeAttributeValue(sid, field, True)&#xd;&#xa;        # iface.messageBar().pushInfo(&quot;[%tile%]&quot;,&quot; combine requested&quot;)&#xd;&#xa;">
      <actionScope id="Layer"/>
      <actionScope id="Feature"/>
      <actionScope id="Canvas"/>
    </actionsetting>
  </attributeactions>
  <layerGeometryType>2</layerGeometryType>
</qgis>
