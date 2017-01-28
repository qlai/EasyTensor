var $ = go.GraphObject.make;  // for conciseness in defining templates

var myDiagram;

function init() {
    
    //define diagram
    myDiagram = $(go.Diagram, "myDiagramDiv",
    {
        
          allowDrop: true,  // must be true to accept drops from the Palette
          "draggingTool.dragsLink": true,
          "draggingTool.isGridSnapEnabled": true,
          "linkingTool.isUnconnectedLinkValid": true,
          "linkingTool.portGravity": 20,
          "relinkingTool.isUnconnectedLinkValid": true,
          "relinkingTool.portGravity": 20,
          "relinkingTool.fromHandleArchetype":
            $(go.Shape, "Diamond", { segmentIndex: 0, cursor: "pointer", desiredSize: new go.Size(8, 8), fill: "tomato", stroke: "darkred" }),
          "relinkingTool.toHandleArchetype":
            $(go.Shape, "Diamond", { segmentIndex: -1, cursor: "pointer", desiredSize: new go.Size(8, 8), fill: "darkred", stroke: "tomato" }),
          "linkReshapingTool.handleArchetype":
            $(go.Shape, "Diamond", { desiredSize: new go.Size(7, 7), fill: "lightblue", stroke: "deepskyblue" }),
          //rotatingTool: $(TopRotatingTool),  // defined below
          "rotatingTool.snapAngleMultiple": 15,
          "rotatingTool.snapAngleEpsilon": 15,
          "undoManager.isEnabled": true
          
    });  

    //used in my
    var nodeSelectionAdornmentTemplate =
      $(go.Adornment, "Auto",
        $(go.Shape, { fill: null, stroke: "deepskyblue", strokeWidth: 1.5, strokeDashArray: [4, 2] }),
        $(go.Placeholder)
      );

    //define palette
    var myPalette = 
      $(go.Palette, "myPaletteDiv",  // must name or refer to the DIV HTML element
        {
          maxSelectionCount: 1,
          nodeTemplateMap: myDiagram.nodeTemplateMap,  // share the templates used by myDiagram
          linkTemplate: // simplify the link template, just in this Palette
            $(go.Link,
              { // because the GridLayout.alignment is Location and the nodes have locationSpot == Spot.Center,
                // to line up the Link in the same manner we have to pretend the Link has the same location spot
                locationSpot: go.Spot.Center,
                selectionAdornmentTemplate:
                  $(go.Adornment, "Link",
                    { locationSpot: go.Spot.Center },
                    $(go.Shape,
                      { isPanelMain: true, fill: null, stroke: "deepskyblue", strokeWidth: 0 }),
                    $(go.Shape,  // the arrowhead
                      { toArrow: "Standard", stroke: null })
                  )
              },
              {
                routing: go.Link.AvoidsNodes,
                curve: go.Link.JumpOver,
                corner: 5,
                toShortLength: 4
              },
              new go.Binding("points"),
              $(go.Shape,  // the link path shape
                { isPanelMain: true, strokeWidth: 2 }),
              $(go.Shape,  // the arrowhead
                { toArrow: "Standard", stroke: null })
            ),
          model: new go.GraphLinksModel([  // specify the contents of the Palette
            { text: "Input", figure: "Circle", fill: "lightgray", para: "Dimension:#" },
            { text: "Output", figure: "Circle", fill: "lightgray", para: "Dimension:#" },
            { text: "Layer", figure: "RoundedRectangle", fill: "lightyellow" , para: "Dimension:#\nActivation: ReLU/ReLU6/CreLU/ELU/Softplus/Softsign/Sigmoid/Tanh\nmulti:#" },   
          ], [
            // the Palette also has a disconnected Link, which the user can drag-and-drop
            { points: new go.List(go.Point).addAll([new go.Point(0, 0), new go.Point(30, 0), new go.Point(30, 40), new go.Point(60, 40)]) }
          ])
        });

    //used in myPalette.nodeTemplate to define the collapse list
    var actionTemplate =
        $(go.Panel, "Horizontal",
          $(go.Shape,
            { width: 12, height: 12 },
            new go.Binding("figure"),
            new go.Binding("fill")
          ),
          $(go.TextBlock,
            { font: "10pt Verdana, sans-serif" },
            new go.Binding("text")
          )
    );

    //define node apperance
    myPalette.nodeTemplate =
      $(go.Node, "Spot",
        { locationSpot: go.Spot.Center },
        new go.Binding("location", "loc", go.Point.parse).makeTwoWay(go.Point.stringify),
        { selectable: true, selectionAdornmentTemplate: nodeSelectionAdornmentTemplate },
        //{ resizable: true, resizeObjectName: "PANEL", resizeAdornmentTemplate: nodeResizeAdornmentTemplate },
        //{ rotatable: true, rotateAdornmentTemplate: nodeRotateAdornmentTemplate },
        new go.Binding("angle").makeTwoWay(),
        // the main object is a Panel that surrounds a TextBlock with a Shape
        $(go.Panel, "Auto",
          { name: "PANEL" },
          //{ isSubGraphExpanded: false },
          new go.Binding("desiredSize", "size", go.Size.parse).makeTwoWay(go.Size.stringify),
          $(go.Shape, "Rectangle",  // default figure
            {
              portId: "", // the default port: if no spot on link data, use closest side
              fromLinkable: true, toLinkable: true, cursor: "pointer",
              fill: "white",  // default color
              strokeWidth: 2
            },
            new go.Binding("figure"),
            new go.Binding("fill")),
          $(go.Panel, "Vertical",
            $(go.TextBlock,
              {
                font: "bold 12pt Helvetica, Arial, sans-serif",
                margin: 8,
                maxSize: new go.Size(160, NaN),
                wrap: go.TextBlock.WrapFit,
                //editable: true
              },
              new go.Binding("text").makeTwoWay()),
            $(go.Panel, "Vertical",
                  { visible: true },  // not visible unless there is more than one action
                  new go.Binding("visible",'para'),
                  /*
                  new go.Binding("visible", "para", function(acts) {
                    return (Array.isArray(acts) && acts.length > 0);
                  }),
                  */
                  // headered by a label and a PanelExpanderButton inside a Table
                  $(go.Panel, "Table",
                    { stretch: go.GraphObject.Horizontal },
                    $(go.TextBlock, "Parameters",
                      { alignment: go.Spot.Left,
                      font: "10pt Verdana, sans-serif"
                      }
                    ),
                    $("PanelExpanderButton", "COLLAPSIBLE",  // name of the object to make visible or invisible
                      { column: 1, alignment: go.Spot.Right, }
                    )
                  ), // end Table panel
                  // with the list data bound in the Vertical Panel
                  $(go.TextBlock,
                    { name: "COLLAPSIBLE",  // identify to the PanelExpanderButton
                      stretch: go.GraphObject.Horizontal,  // take up whole available width
                      background: "white",  // to distinguish from the node's body
                      editable: true
                    },
                    new go.Binding("text", "para")  // bind TextBlock.text to nodedata.para
                  )  // end action list Vertical Panel
              )  // end optional Vertical Panel
          )
        ),
        
        // four small named ports, one on each side:
        makePort("T", go.Spot.Top, false, true),
        makePort("L", go.Spot.Left, true, true),
        makePort("R", go.Spot.Right, true, true),
        makePort("B", go.Spot.Bottom, true, false),
        { // handle mouse enter/leave events to show/hide the ports
          mouseEnter: function(e, node) { showSmallPorts(node, true); },
          mouseLeave: function(e, node) { showSmallPorts(node, false); }
        }
      );
    

    /*
    myDiagram.groupTemplate =
    $(go.Group, "Auto",
      { layout: $(go.LayeredDigraphLayout,
                  { direction: 0, columnSpacing: 10 }) },
      { isSubGraphExpanded: false },
      $(go.Shape, "RoundedRectangle", // surrounds everything
        { parameter1: 10}),
      $(go.Panel, "Vertical",  // position header above the subgraph
        { defaultAlignment: go.Spot.Left },
        $(go.Panel, "Horizontal",  // the header
          { defaultAlignment: go.Spot.Top },
          $("SubGraphExpanderButton"),  // this Panel acts as a Button
          $(go.TextBlock,     // group title near top, next to button
            { font: "Bold 12pt Sans-Serif" },
            new go.Binding("text", "key"))
        ),
        $(go.Placeholder,     // represents area for all member parts
          { padding: new go.Margin(0, 10), background: "white" })
      )


      // four small named ports, one on each side:
        makePort("T", go.Spot.Top, false, true),
        makePort("L", go.Spot.Left, true, true),
        makePort("R", go.Spot.Right, true, true),
        makePort("B", go.Spot.Bottom, true, false),
        { // handle mouse enter/leave events to show/hide the ports
          mouseEnter: function(e, node) { showSmallPorts(node, true); },
          mouseLeave: function(e, node) { showSmallPorts(node, false); }
        }
    );
*/


    myDiagram.linkTemplate =
      $(go.Link,  // the whole link panel
        //{ selectable: true, selectionAdornmentTemplate: linkSelectionAdornmentTemplate },
        //{ relinkableFrom: true, relinkableTo: true, reshapable: true },
        {
          routing: go.Link.AvoidsNodes,
          curve: go.Link.JumpOver,
          corner: 5,
          toShortLength: 4
        },
        new go.Binding("points").makeTwoWay(),
        $(go.Shape,  // the link path shape
          { isPanelMain: true, strokeWidth: 2 }),
        $(go.Shape,  // the arrowhead
          { toArrow: "Standard", stroke: null }),
        $(go.Panel, "Auto",
          new go.Binding("visible", "isSelected").ofObject(),
          $(go.Shape, "RoundedRectangle",  // the link shape
            { fill: "#F8F8F8", stroke: null }),
          $(go.TextBlock,
            {
              textAlign: "center",
              font: "10pt helvetica, arial, sans-serif",
              stroke: "#919191",
              margin: 2,
              minSize: new go.Size(10, NaN),
              editable: true
            },
            new go.Binding("text").makeTwoWay())
        )
    );

    // Define a function for creating a "port" that is normally transparent.
    // The "name" is used as the GraphObject.portId, the "spot" is used to control how links connect
    // and where the port is positioned on the node, and the boolean "output" and "input" arguments
    // control whether the user can draw links from or to the port.
    function makePort(name, spot, output, input) {
      // the port is basically just a small transparent square
      return $(go.Shape, "Circle",
               {
                  fill: null,  // not seen, by default; set to a translucent gray by showSmallPorts, defined below
                  stroke: null,
                  desiredSize: new go.Size(7, 7),
                  alignment: spot,  // align the port on the main Shape
                  alignmentFocus: spot,  // just inside the Shape
                  portId: name,  // declare this object to be a "port"
                  fromSpot: spot, toSpot: spot,  // declare where links may connect at this port
                  fromLinkable: output, toLinkable: input,  // declare whether the user may draw links to/from here
                  cursor: "pointer"  // show a different cursor to indicate potential link point
               });
    }

    function showSmallPorts(node, show) {
      node.ports.each(function(port) {
        if (port.portId !== "") {  // don't change the default port, which is the big shape
          port.fill = show ? "rgba(0,0,0,.3)" : null;
        }
      });
    }

}

function toJson() {
    var modelDescription = myDiagram.model.toJson();
    console.log(modelDescription);
    /*
    var socket = new WebSocket('ws://localhost:5000');
    
    socket.addEventListener('open', function (event) {
    socket.send('Connected successfully!');
    });

    socket.addEventListener('message', function (event) {
      console.log('Message from server', event.data);
    });

    socket.send(JSON.stringify(modelDescription));
    */
    var tmp = document.getElementsByName("ModelDescription")[0];
    tmp.value = stringify(modelDescription);
}



    

          

